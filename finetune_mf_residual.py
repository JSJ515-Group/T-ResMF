# finetune_mf_residual.py
import os
import math
import pandas as pd
import setproctitle
import importlib
from utils.trainer import Trainer
from utils.logger import Logger, log_exceptions
import torch
import torch.nn.functional as F
import numpy as np
import random
from utils.dataloader import EdgeListData
from utils.parse_args import args
from os import path
import datetime
import sys

sys.path.append("./")

setproctitle.setproctitle("GraphProMFResFinetune")
args.phase = "finetune"
modules_class = "modules."
if args.plugin:
    modules_class = "modules.plugins."


def import_model():
    module = importlib.import_module(modules_class + args.model)
    return getattr(module, args.model)


def import_pretrained_model():
    module = importlib.import_module(modules_class + args.pre_model)
    return getattr(module, args.pre_model)


def import_finetune_model():
    module = importlib.import_module(modules_class + args.f_model)
    return getattr(module, args.f_model)


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _extract_base_emb(sd, prefix):
    """
    兼容：
      - user_embedding.base_embedding.weight（GraphProCCF）
      - user_embedding.weight
      - user_embedding
    """
    if prefix in sd:
        return sd[prefix]
    candidates = [k for k in sd.keys() if k.startswith(prefix) and k.endswith("weight")]
    if len(candidates) == 0:
        raise KeyError(f"在 state_dict 中找不到以 {prefix} 开头的 embedding 参数")
    candidates = sorted(candidates, key=lambda x: (("base_embedding" not in x), len(x)))
    return sd[candidates[0]]


def _load_pisa_weight(pisa_dir: str, stage: int, kind: str, lag: int, num_users: int):
    """
    kind:
      - "vs_pretrain"  -> user_align_weight_stage_{stage}_vs_pretrain.npy
      - "lag"          -> user_align_weight_stage_{stage}_lag_{lag}.npy  (lag==1 允许 fallback stage_{stage}.npy)
    """
    if kind == "vs_pretrain":
        cand = [os.path.join(pisa_dir, f"user_align_weight_stage_{stage}_vs_pretrain.npy")]
    else:
        cand = [os.path.join(pisa_dir, f"user_align_weight_stage_{stage}_lag_{lag}.npy")]
        if lag == 1:
            cand.append(os.path.join(pisa_dir, f"user_align_weight_stage_{stage}.npy"))

    for p in cand:
        if os.path.isfile(p):
            w = np.load(p).astype(np.float32).reshape(-1)
            if w.shape[0] != num_users:
                raise RuntimeError(f"PISA weight size mismatch: got {w.shape[0]}, expect {num_users}, file={p}")
            return torch.from_numpy(w).float(), p
    return None, None


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _compute_base_weights(window_k: int, scheme: str, pretrain_weight: float, device):
    """
    返回 base_w: Tensor [1+window_k]
    - scheme=graphpro：pre + (1-pre)*[k, k-1, ..., 1]/sum
    - scheme=recent：只用上一个阶段（不含 pre）
    """
    if scheme == "recent":
        # 强制只用 lag1
        base = torch.tensor([0.0, 1.0], dtype=torch.float32, device=device)
        return base

    # graphpro
    pre_w = float(pretrain_weight)
    if window_k <= 0:
        return torch.tensor([1.0], dtype=torch.float32, device=device)

    hist = torch.arange(window_k, 0, -1, dtype=torch.float32, device=device)  # [k,k-1,...,1]
    hist = hist / (hist.sum() + 1e-12)  # sum=1
    hist = (1.0 - pre_w) * hist
    base = torch.cat([torch.tensor([pre_w], device=device), hist], dim=0)
    return base  # [1+k]


def _auto_scale(sims_centered, eta, target_ratio, q, max_scale):
    """
    sims_centered: (U,K) 已经 per-user 去均值后的 sims
    目标：让“典型差异”能把 softmax 推到 target_ratio（比如 0.55/0.45）
    """
    target = _logit(target_ratio)  # ~0.2007 for 0.55
    U, K = sims_centered.shape

    if K == 2:
        diff = (sims_centered[:, 0] - sims_centered[:, 1]).abs()
        denom = torch.quantile(diff, q).clamp_min(1e-8)
    else:
        spread = sims_centered.abs().max(dim=1).values  # 每个用户的“最大偏移量”
        denom = torch.quantile(spread, q).clamp_min(1e-8)

    scale = float(target) / float(float(eta) * float(denom) + 1e-12)
    scale = max(1.0, min(scale, float(max_scale)))
    return scale


def pisa_mix_weights(
    sims_raw,           # (U,K) in [0,1]
    base_w,             # (K,)
    eta=0.35,
    eps=0.02,
    use_delta=True,
    auto_scale=True,
    target_ratio=0.55,
    scale_q=0.5,
    scale_max=2000.0,
    sim_clip=4.0,
    invalid_fill="stage_mean",  # none / stage_mean
    min_dev_invalid=0.02,
    invalid_alpha=0.2,
    logger=None,
    stage=None,
    pisa_files=None,
):
    """
    PISA 只“影响原本 base_w”，不是替代 base_w：
      logits = log(base_w + eps) + eta * (scaled centered sims)
      w = softmax(logits)

    invalid 用户（全 0）也会跟着偏一点：
      - 先用 stage_mean 填充 sims（可选）
      - 再把 invalid 的权重做一个“小幅偏移”（min_dev_invalid）
    """
    device = sims_raw.device
    U, K = sims_raw.shape

    sims = sims_raw.clone()

    # invalid：所有列都为 0
    invalid = (sims.sum(dim=1) <= 1e-12)

    # 让 invalid 的 sims 也有“全局趋势”
    if invalid_fill == "stage_mean" and (~invalid).any():
        col_mean = sims[~invalid].mean(dim=0)
        sims[invalid] = col_mean.view(1, K)

    # 去掉 per-user 偏置：只保留“相对更像谁”
    if use_delta:
        sims = sims - sims.mean(dim=1, keepdim=True)

    # 自动放大：让你能看到 0.45/0.55 这种变化
    sim_scale = 1.0
    if auto_scale:
        sim_scale = _auto_scale(sims, eta=eta, target_ratio=target_ratio, q=scale_q, max_scale=scale_max)

    sims_scaled = torch.clamp(sims * sim_scale, -sim_clip, sim_clip)

    base_w = base_w / (base_w.sum() + 1e-12)
    logits = torch.log(base_w + eps).view(1, K) + float(eta) * sims_scaled
    w = torch.softmax(logits, dim=1)  # (U,K)

    # --- invalid 也动一点，但不要太大 ---
    if invalid.any() and min_dev_invalid > 0:
        # 用全局“哪个更像”决定方向：看 sims_scaled 的列均值
        global_pref = sims_scaled.mean(dim=0)  # (K,)
        j_star = int(torch.argmax(global_pref).item())

        # invalid 的默认：base_w 混一点 w_mean
        w_mean = w.mean(dim=0)
        w_inv = (1.0 - float(invalid_alpha)) * base_w + float(invalid_alpha) * w_mean
        w_inv = w_inv / (w_inv.sum() + 1e-12)

        # 强制有最小偏移：把 j_star 增加一点点（从 pretrain 那一列扣一些）
        w_inv2 = w_inv.clone()
        inc = float(min_dev_invalid)
        if K == 2:
            # 二分类：直接把 pretrain 那列按方向推一点
            # j_star==0 表示更像 pretrain；j_star==1 表示更像 lag1
            if j_star == 0:
                w0 = float(torch.clamp(w_inv2[0] + inc, 0.0, 1.0))
            else:
                w0 = float(torch.clamp(w_inv2[0] - inc, 0.0, 1.0))
            w_inv2[0] = w0
            w_inv2[1] = 1.0 - w0
        else:
            # 多分类：给 j_star +inc，从 pretrain(0) 扣（如果 j_star=0 就从 lag1 扣）
            src = 0 if j_star != 0 else 1
            take = min(inc, float(w_inv2[src]))
            w_inv2[j_star] = w_inv2[j_star] + take
            w_inv2[src] = w_inv2[src] - take
            w_inv2 = torch.clamp(w_inv2, min=0.0)
            w_inv2 = w_inv2 / (w_inv2.sum() + 1e-12)

        w[invalid] = w_inv2.view(1, K)

    # --- 日志：打印确切比例 ---
    if logger is not None:
        base_list = [float(x) for x in base_w.detach().cpu().tolist()]
        mean_list = [float(x) for x in w.mean(dim=0).detach().cpu().tolist()]

        # 打印 pretrain 权重分布（最常关注）
        pre = w[:, 0].detach()
        qv = [float(torch.quantile(pre, t).item()) for t in [0.0, 0.5, 0.9, 0.99, 1.0]]

        logger.info(f"[Interp-base] stage={stage} window_k={K-1} scheme={str(getattr(args,'interp_scheme','graphpro'))} base_w={base_list}")
        logger.info(
            f"[PISA-mix] stage={stage} window_k={K-1} "
            f"mean_w={mean_list} "
            f"pre(min/p50/p90/p99/max)=({qv[0]:.4f},{qv[1]:.4f},{qv[2]:.4f},{qv[3]:.4f},{qv[4]:.4f}) "
            f"eta={eta} eps={eps} scale={sim_scale:.3f} "
            f"target={target_ratio} q={scale_q} "
            f"invalid_ratio={float(invalid.float().mean().item()):.6f} "
            f"files={pisa_files}"
        )

        # 打几个样本 + 极端用户（避免 user0/user1 一直 0.5 看不出差）
        sample_ids = [0, 1, 2, 10, 100]
        for uid in sample_ids:
            if uid < U:
                ww = [float(x) for x in w[uid].detach().cpu().tolist()]
                logger.info(f"[PISA-mix-sample] stage={stage} user={uid} (pre + lags)={ww}")

        imin = int(torch.argmin(pre).item())
        imax = int(torch.argmax(pre).item())
        logger.info(f"[PISA-mix-extreme] stage={stage} argmin_user={imin} w_pre={float(pre[imin]):.4f}")
        logger.info(f"[PISA-mix-extreme] stage={stage} argmax_user={imax} w_pre={float(pre[imax]):.4f}")

        if invalid.any():
            inv_u = int(torch.where(invalid)[0][0].item())
            ww = [float(x) for x in w[inv_u].detach().cpu().tolist()]
            logger.info(f"[PISA-mix-invalid-sample] stage={stage} user={inv_u} (pre + lags)={ww}")

    return w  # (U,K)


init_seed(args.seed)
logger = Logger(args)

pretrain_data = path.join(args.data_path, "pretrain.txt")
pretrain_val_data = path.join(args.data_path, "pretrain_val.txt")
finetune_data = path.join(args.data_path, "fine_tune.txt")

# 自动检测 test_i
test_files = []
for i in range(1, 100):
    fp = path.join(args.data_path, f"test_{i}.txt")
    if os.path.isfile(fp):
        test_files.append(fp)
    else:
        break
test_data_num = len(test_files)
logger.log(f"test_data_num: {test_data_num}")

all_data = [pretrain_data, finetune_data, *test_files]
recalls, ndcgs = [], []


@log_exceptions
def run():
    from modules.plugins.GraphProMFResFinetune import GraphProMFResFinetune

    pretrain_dataset = EdgeListData(pretrain_data, pretrain_val_data)

    # 0) 预训练 embedding X_p
    pre_state = torch.load(args.pre_model_path, map_location="cpu")
    pre_user_emb_cpu = _extract_base_emb(pre_state, "user_embedding").detach().cpu()
    pre_item_emb_cpu = _extract_base_emb(pre_state, "item_embedding").detach().cpu()

    history_user_embs = []  # X_1, X_2... (CPU)
    history_item_embs = []  # X_1, X_2... (CPU)

    # === 参数 ===
    pretrain_weight = float(getattr(args, "pretrain_weight", 0.5))
    init_l2norm = bool(getattr(args, "init_l2norm", True))
    interp_scheme = str(getattr(args, "interp_scheme", "graphpro"))  # graphpro / recent

    use_pisa_mix = bool(getattr(args, "use_pisa_mix", 0))
    pisa_dir = str(getattr(args, "pisa_dir", os.path.join(args.data_path, "pisa_pref_weights")))
    pisa_strict = bool(getattr(args, "pisa_strict", 0))

    pisa_eta_mix = float(getattr(args, "pisa_eta_mix", 0.35))
    pisa_eps = float(getattr(args, "pisa_eps", 0.02))
    pisa_use_delta = bool(getattr(args, "pisa_use_delta", 1))

    pisa_auto_scale = bool(getattr(args, "pisa_auto_scale", 1))
    pisa_target_ratio = float(getattr(args, "pisa_target_ratio", 0.55))
    pisa_scale_q = float(getattr(args, "pisa_scale_q", 0.5))
    pisa_scale_max = float(getattr(args, "pisa_scale_max", 2000.0))
    pisa_sim_clip = float(getattr(args, "pisa_sim_clip", 4.0))

    pisa_invalid_fill = str(getattr(args, "pisa_invalid_fill", "stage_mean"))  # none/stage_mean
    pisa_min_dev_invalid = float(getattr(args, "pisa_min_dev_invalid", 0.02))
    pisa_invalid_alpha = float(getattr(args, "pisa_invalid_alpha", 0.2))

    for num_stage in range(1, test_data_num + 1):
        interval = int(getattr(args, "updt_inter", 1))
        k_hist = len(history_user_embs)

        # -------- 1) 初始化 embedding --------
        if k_hist == 0:
            # 第一阶段：纯预训练
            init_user_emb = pre_user_emb_cpu.to(args.device)
            init_item_emb = pre_item_emb_cpu.to(args.device)
            logger.info(f"[Stage {num_stage}] init=pre_only (no history)")
        else:
            # recent 模式只用 lag1
            if interp_scheme == "recent":
                window_k = 1
            else:
                window_k = min(interval, k_hist)

            base_user = pre_user_emb_cpu.to(args.device)
            base_item = pre_item_emb_cpu.to(args.device)

            # recent list 按 lag 顺序：[X_{t-1}, X_{t-2}, ...]
            recent_user_list = [history_user_embs[-lag].to(args.device) for lag in range(1, window_k + 1)]
            recent_item_list = [history_item_embs[-lag].to(args.device) for lag in range(1, window_k + 1)]

            # base 权重（原 GraphPro 比例）
            base_w = _compute_base_weights(window_k, interp_scheme, pretrain_weight, device=args.device)  # (1+k)
            base_w_list = [float(x) for x in base_w.detach().cpu().tolist()]
            logger.info(f"[Interp-base] stage={num_stage} window_k={window_k} scheme={interp_scheme} base_w={base_w_list}")

            # --- PISA mix：用当前阶段与 pretrain / 各 lag 的相似度来“影响 base_w” ---
            pisa_ok = False
            pisa_files = []

            if use_pisa_mix:
                if not os.path.isdir(pisa_dir):
                    if pisa_strict:
                        raise RuntimeError(f"pisa_dir not found: {pisa_dir}")
                else:
                    U = base_user.shape[0]
                    sim_cols = []

                    # col0: vs_pretrain
                    w0, p0 = _load_pisa_weight(pisa_dir, num_stage, kind="vs_pretrain", lag=0, num_users=U)
                    if w0 is not None:
                        sim_cols.append(w0.to(args.device))
                        pisa_files.append(os.path.basename(p0))

                        # cols: lag1..lagK
                        ok = True
                        for lag in range(1, window_k + 1):
                            wi, pi = _load_pisa_weight(pisa_dir, num_stage, kind="lag", lag=lag, num_users=U)
                            if wi is None:
                                ok = False
                                break
                            sim_cols.append(wi.to(args.device))
                            pisa_files.append(os.path.basename(pi))

                        pisa_ok = ok

            if use_pisa_mix and pisa_ok:
                sims = torch.stack(sim_cols, dim=1)  # (U, 1+k)

                w_u = pisa_mix_weights(
                    sims_raw=sims,
                    base_w=base_w,
                    eta=pisa_eta_mix,
                    eps=pisa_eps,
                    use_delta=pisa_use_delta,
                    auto_scale=pisa_auto_scale,
                    target_ratio=pisa_target_ratio,
                    scale_q=pisa_scale_q,
                    scale_max=pisa_scale_max,
                    sim_clip=pisa_sim_clip,
                    invalid_fill=pisa_invalid_fill,
                    min_dev_invalid=pisa_min_dev_invalid,
                    invalid_alpha=pisa_invalid_alpha,
                    logger=logger,
                    stage=num_stage,
                    pisa_files=pisa_files,
                )  # (U,1+k)

                # user embedding：per-user 权重
                init_user_emb = w_u[:, 0].unsqueeze(1) * base_user
                for i in range(window_k):
                    init_user_emb = init_user_emb + w_u[:, i + 1].unsqueeze(1) * recent_user_list[i]

                # item embedding：没有 per-item PISA，用 user 权重均值（stage-level）
                w_item = w_u.mean(dim=0)  # (1+k)
                init_item_emb = w_item[0] * base_item
                for i in range(window_k):
                    init_item_emb = init_item_emb + w_item[i + 1] * recent_item_list[i]
            else:
                if use_pisa_mix and pisa_strict:
                    raise RuntimeError(f"PISA files missing at stage={num_stage}, need window_k={window_k}, pisa_dir={pisa_dir}")
                # fallback：纯 base_w 插值
                init_user_emb = base_w[0] * base_user
                init_item_emb = base_w[0] * base_item
                for i in range(window_k):
                    init_user_emb = init_user_emb + base_w[i + 1] * recent_user_list[i]
                    init_item_emb = init_item_emb + base_w[i + 1] * recent_item_list[i]
                logger.info(f"[Stage {num_stage}] PISA missing -> fallback base weights, window_k={window_k}")

            if init_l2norm:
                init_user_emb = F.normalize(init_user_emb, dim=1)
                init_item_emb = F.normalize(init_item_emb, dim=1)

        # -------- 2) 数据集 --------
        test_data_idx = num_stage + 1
        ft_data_idx = test_data_idx - 1
        logger.info(
            f"Finetune Stage {num_stage}, "
            f"test data: {all_data[test_data_idx]}, "
            f"finetune data {all_data[ft_data_idx]}"
        )

        finetune_dataset = EdgeListData(
            train_file=all_data[ft_data_idx],
            test_file=path.join(args.data_path, f"test_{num_stage}.txt"),
            phase="finetune",
            pre_dataset=pretrain_dataset,
            has_time=True,
            user_hist_files=all_data[:ft_data_idx],
        )

        # -------- 3) 模型 --------
        model = GraphProMFResFinetune(
            finetune_dataset,
            pretrained_user_emb=init_user_emb,
            pretrained_item_emb=init_item_emb,
            phase="finetune",
        ).to(args.device)

        trainer = Trainer(finetune_dataset, logger, pre_dataset=pretrain_dataset)
        best_perform = trainer.train_finetune(model)

        recalls.append(best_perform["recall"][0])
        ndcgs.append(best_perform["ndcg"][0])

        # -------- 4) 存历史 embedding --------
        with torch.no_grad():
            final_user_emb, final_item_emb = model.generate()
            history_user_embs.append(final_user_emb.detach().cpu())
            history_item_embs.append(final_item_emb.detach().cpu())

        args.exp_time = datetime.datetime.now().strftime("%b-%d-%Y_%H-%M-%S")

    logger.info(
        f"recalls: {recalls} \n"
        f"ndcgs: {ndcgs} \n"
        f"avg. recall: {np.round(np.mean(recalls), 4)}, "
        f"avg. ndcg: {np.round(np.mean(ndcgs), 4)}"
    )


if __name__ == "__main__":
    run() 
