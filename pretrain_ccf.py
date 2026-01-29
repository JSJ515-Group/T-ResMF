# pretrain.py
"""
Usage example:
python pretrain.py --data_path dataset/taobao --exp_name pretrain_ccf --phase pretrain \
    --device cuda:0 --encoder MF --lr 1e-3 --batch_size 2048 --num_epochs 300 \
    --use_ccf --use_bpr_with_cl --ccf_lambda 0.5 --ssl_lambda 5.0 --lambda_t 0.05 \
    --edge_dropout 0.5 --hour_interval_pre 24 --save_path saved/taobao/pretrain_ccf
"""
import os
import time
import torch
import random
import numpy as np
from os import path, makedirs
from tqdm import tqdm

from utils.parse_args import args
from utils.logger import Logger
from utils.metrics import Metric
from utils.dataloader import GraphProCCFData  # 你的数据类（之前实现）
# import 模型（你之前写的预训练模型文件名）
from modules.TResMF import TResMF  # 如果类/文件名不同，按实际改

# -------------------------
# helpers
# -------------------------
def ensure_dir(d):
    if not path.exists(d):
        makedirs(d, exist_ok=True)

@torch.no_grad()
def evaluate_batched(model, dataset, k=20, eval_batch_size=512, device='cpu'):
    """
    分块评估 recall@k 和 ndcg@k：
    - 生成全量 user/item embedding via model.generate()
    - 对测试用户集合（dataset.test_user_dict）按 eval_batch_size 分块计算 topk
    - 屏蔽训练历史中的物品（train_user_dict）以避免泄漏
    """
    model.eval()
    user_emb_all, item_emb_all = model.generate()  # tensors on device (model should return in correct device)
    # ensure on device
    user_emb_all = user_emb_all.to(device)
    item_emb_all = item_emb_all.to(device)

    test_users = list(dataset.test_user_dict.keys())
    if len(test_users) == 0:
        return {'recall':[0.0], 'ndcg':[0.0]}

    recalls = []
    ndcgs = []
    K = k

    # precompute ideal DCG denominators for possible gt sizes up to K
    import math
    def dcg_at_k(hit_positions):
        return sum([1.0 / math.log2(p + 2) for p in hit_positions])  # p is index 0-based

    for start in range(0, len(test_users), eval_batch_size):
        end = min(len(test_users), start + eval_batch_size)
        u_ids = test_users[start:end]
        u_idx = torch.LongTensor(u_ids).to(device)
        u_emb = user_emb_all[u_idx]  # (B, D)

        # score matrix for this batch: (B, num_items)
        scores = torch.matmul(u_emb, item_emb_all.t())  # (B, I)

        # mask training items
        for bi, u in enumerate(u_ids):
            train_items = dataset.train_user_dict.get(int(u), [])
            if train_items:
                scores[bi, train_items] = -1e9

        # topk indices
        topk_vals, topk_idx = torch.topk(scores, K, dim=1)
        topk_idx = topk_idx.cpu().numpy()  # (B, K)

        # for each user compute recall & ndcg
        for i, u in enumerate(u_ids):
            gt = dataset.test_user_dict.get(int(u), [])
            if not gt:
                # skip users with no groundtruth
                continue
            gt_set = set(int(x) for x in gt)
            pred = topk_idx[i].tolist()
            hits = [1 if p in gt_set else 0 for p in pred]
            num_hits = sum(hits)
            recall_u = num_hits / min(len(gt_set), K)
            # ndcg
            hit_positions = [pos for pos, h in enumerate(hits) if h]
            dcg = dcg_at_k(hit_positions)
            # ideal dcg: assume all gt items are in top positions
            ideal_len = min(len(gt_set), K)
            ideal_dcg = sum([1.0 / math.log2(p + 2) for p in range(ideal_len)])
            ndcg_u = (dcg / ideal_dcg) if ideal_dcg > 0 else 0.0
            recalls.append(recall_u)
            ndcgs.append(ndcg_u)

    # mean
    recall_mean = float(np.mean(recalls)) if recalls else 0.0
    ndcg_mean = float(np.mean(ndcgs)) if ndcgs else 0.0
    return {'recall':[recall_mean], 'ndcg':[ndcg_mean]}

# -------------------------
# training
# -------------------------
def train():
    device = torch.device(args.device if args.device else 'cpu')
    logger = Logger(args)

    # load dataset (GraphProCCFData expects args and phase)
    dataset = GraphProCCFData(args, phase='pretrain')

    # instantiate model: uses your pretrain model class (GraphProCCF)
    model = TResMF(dataset, phase='pretrain').to(device)

    # ensure model has rating (if not, add simple one)
    if not hasattr(model, 'rating'):
        # attach rating at runtime
        def rating_fn(u_emb, i_emb):
            return torch.matmul(u_emb, i_emb.t())
        model.rating = rating_fn  # monkey patch (optional)
        print("Warning: model had no rating(); monkey-patched simple dot product.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    num_steps_per_epoch = int(np.ceil(dataset.num_users / max(1, args.batch_size)))  # fallback to users-based
    # We'll use dataset.get_contrastive_batch to sample LightCCF-style batches
    # If your dataset has many users, use get_contrastive_batch for user-based CL sampling.

    best_recall = 0.0
    save_dir = path.join(getattr(args, 'save_path', args.save_dir if hasattr(args, 'save_dir') else 'saved'), args.exp_name)
    ensure_dir(save_dir)

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        iters = 0
        t0 = time.time()

        # We'll iterate over number of steps = num_edges // batch_size roughly,
        # but prefer using get_contrastive_batch repeatedly until we accumulate enough updates
        num_updates = max(1, dataset.num_edges // args.batch_size)

        pbar = tqdm(range(num_updates), desc=f"[Epoch {epoch}/{args.num_epochs}]")
        for _ in pbar:
            # sample a contrastive batch (LightCCF style)
            users_b, pos_b, neg_b = dataset.get_contrastive_batch(args.batch_size, num_negatives=1)
            # users_b: (B,), pos_b: (B,), neg_b: (B,) OR (B*num_negatives,)
            if users_b.numel() == 0:
                continue
            users_b = users_b.to(device)
            pos_b = pos_b.to(device)

            # convert neg_b to same shape: get_contrastive_batch returns num_negatives flattened
            if neg_b.numel() == users_b.size(0):
                neg_b_used = neg_b.to(device)
            else:
                # if multiple negatives, take first per user
                neg_b_used = neg_b.view(users_b.size(0), -1)[:, 0].to(device)

            optimizer.zero_grad()
            loss, loss_dict = model.cal_loss((users_b, pos_b, neg_b_used))
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            iters += 1
            pbar.set_postfix({'loss': f"{(epoch_loss/iters):.6f}", 'bpr_loss': f"{loss_dict.get('rec_loss', 0):.4f}", 'cl_loss': f"{loss_dict.get('cl_loss', 0):.4f}"})

        t1 = time.time()
        # evaluate after epoch (use batched eval to avoid OOM)
        eval_res = evaluate_batched(model, dataset, k=int(args.metrics_k), eval_batch_size=getattr(args, 'eval_batch_size', 512), device=device)
        recall_epoch = eval_res['recall'][0]
        ndcg_epoch = eval_res['ndcg'][0]

        # log similar to original
        print(f"[Epoch {epoch} / {args.num_epochs} Training Time: {round(t1 - t0,2)}s ] " +
              f"loss: {(epoch_loss / max(1, iters)):.6f}  recall@{args.metrics_k}: {recall_epoch:.4f} ndcg@{args.metrics_k}: {ndcg_epoch:.4f}")

        # save best
        if recall_epoch > best_recall:
            best_recall = recall_epoch
            save_path = path.join(save_dir, f"saved_model_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Find better model at epoch: {epoch}: recall={best_recall:.4f}  saved to {save_path}")

    print("Training finished. Best recall:", best_recall)


if __name__ == "__main__":
    train()
