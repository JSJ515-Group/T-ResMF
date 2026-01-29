import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.base_model import BaseModel
from utils.parse_args import args

init = nn.init.xavier_uniform_


class GraphProMFResFinetune(BaseModel):
    """
    微调阶段：针对 MF / GraphProCCF 预训练嵌入的残差式微调模型（带 PISA-style 对齐正则）
    - 预训练：MF + CL（GraphProCCF）
    - 微调：base(冻结) + delta(可学习) + BPR + 可选 CL
    - 稳定对齐：对当前 user embedding 与上一阶段 embedding 之间做带权 L2 对齐
        * 权重 λ_su 来自 PISA-style 偏好漂移估计（离线脚本 build_pisa_pref_weights.py）
    """

    def __init__(self, dataset,
                 pretrained_user_emb,
                 pretrained_item_emb,
                 prev_user_emb=None,
                 user_align_weight=None,
                 phase="finetune"):
        super().__init__(dataset)
        self.dataset = dataset
        self.phase = phase
        self.device = args.device

        # 基本尺寸
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.embedding_size = pretrained_user_emb.size(1)

        # 正则 & 时间衰减超参
        self.reg_lambda = float(getattr(args, "reg_lambda", 1e-4))
        self.use_time_decay = bool(getattr(args, "use_time_decay", False))
        self.time_lambda = float(getattr(args, "time_decay_lambda", 0.01))

        # 对比学习（微调阶段）配置
        self.use_cl_ft = bool(getattr(args, "finetune_use_cl", True))
        self.ssl_lambda = float(getattr(args, "ft_ssl_lambda", 2.0))
        self.tau = float(getattr(args, "tau", 0.2))

        # 对齐正则（PISA-style 稳定权重）
        self.align_lambda = float(getattr(args, "ft_align_lambda", 0.0))

        # ====== 1. 冻结的 base embedding（长期偏好） ======
        self.base_user_emb = nn.Embedding(self.num_users, self.embedding_size)
        self.base_item_emb = nn.Embedding(self.num_items, self.embedding_size)
        self.base_user_emb.weight.data.copy_(pretrained_user_emb.detach().clone())
        self.base_item_emb.weight.data.copy_(pretrained_item_emb.detach().clone())
        self.base_user_emb.weight.requires_grad = False
        self.base_item_emb.weight.requires_grad = False

        # ====== 2. 可学习残差 embedding（短期漂移） ======
        self.delta_user_emb = nn.Embedding(self.num_users, self.embedding_size)
        self.delta_item_emb = nn.Embedding(self.num_items, self.embedding_size)
        nn.init.zeros_(self.delta_user_emb.weight)
        nn.init.zeros_(self.delta_item_emb.weight)

        # ====== 3. 门控标量 ======
        self.user_gate_raw = nn.Parameter(torch.tensor(0.1))
        self.item_gate_raw = nn.Parameter(torch.tensor(0.1))

        # ====== 4. 上一阶段 user embedding & PISA-style 对齐权重 ======
        if (prev_user_emb is not None) and (self.align_lambda > 0):
            prev_user_emb = prev_user_emb.to(self.device)
            self.register_buffer("prev_user_emb", prev_user_emb)
        else:
            self.prev_user_emb = None

        if (user_align_weight is not None) and (self.align_lambda > 0):
            if isinstance(user_align_weight, torch.Tensor):
                w = user_align_weight
            else:
                import numpy as np
                assert isinstance(user_align_weight, np.ndarray)
                w = torch.from_numpy(user_align_weight)
            w = w.to(self.device).float()
            self.register_buffer("user_align_weight", w)
        else:
            self.user_align_weight = None

        print("=== GraphProMFResFinetune 初始化（带 CL + PISA-style 对齐正则） ===")
        print(f"用户数: {self.num_users}, 物品数: {self.num_items}")
        print(f"Embedding 维度: {self.embedding_size}")
        print(f"使用时间衰减: {self.use_time_decay}, λ={self.time_lambda}")
        print(f"微调阶段使用CL: {self.use_cl_ft}, ssl_lambda={self.ssl_lambda}, tau={self.tau}")
        print(f"训练期稳定对齐正则: lambda_align={self.align_lambda}")
        if self.prev_user_emb is None or self.user_align_weight is None or self.align_lambda <= 0:
            print("对齐正则当前阶段未启用（缺少 prev_user_emb 或 user_align_weight 或 lambda_align<=0）")
        else:
            nonzero_ratio = float((self.user_align_weight > 0).float().mean().item())
            print(f"对齐正则启用：有效用户比例={nonzero_ratio:.4f}")
        print("==============================================")

    # ---------- 时间衰减权重 ----------
    def compute_time_weight(self, pos_times):
        if (pos_times is None) or (not self.use_time_decay):
            return None

        times = pos_times.view(-1).float().to(self.device)
        t_min = times.min()
        t_max = times.max()

        if (t_max - t_min) < 1e-6:
            return torch.ones_like(times, device=self.device)

        norm_t = (times - t_min) / (t_max - t_min + 1e-8)
        w = torch.exp(-self.time_lambda * (1.0 - norm_t))
        w = torch.clamp(w, min=0.1, max=1.0)
        return w  # [B]

    # ---------- forward ----------
    def forward(self, users, pos_items, neg_items, pos_times=None):
        users = users.long().to(self.device)
        pos_items = pos_items.long().to(self.device)
        neg_items = neg_items.long().to(self.device)
        if pos_times is not None:
            pos_times = pos_times.to(self.device)

        batch_size = users.size(0)

        user_gate = torch.sigmoid(self.user_gate_raw)
        item_gate = torch.sigmoid(self.item_gate_raw)

        u_base = self.base_user_emb(users)
        p_base = self.base_item_emb(pos_items)
        n_base = self.base_item_emb(neg_items)

        u_delta = self.delta_user_emb(users)
        p_delta = self.delta_item_emb(pos_items)
        n_delta = self.delta_item_emb(neg_items)

        u_vec = u_base + user_gate * u_delta
        p_vec = p_base + item_gate * p_delta
        n_vec = n_base + item_gate * n_delta

        # -------- BPR：不加时间权重，直接平均 --------
        pos_scores = torch.sum(u_vec * p_vec, dim=1)
        neg_scores = torch.sum(u_vec * n_vec, dim=1)
        per_sample_bpr = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-12)
        bpr_loss = per_sample_bpr.mean()

        # -------- 残差正则 --------
        reg_loss = 0.5 * (
            u_delta.norm(2).pow(2) +
            p_delta.norm(2).pow(2) +
            n_delta.norm(2).pow(2)
        ) / users.shape[0] * self.reg_lambda

        # -------- 对比学习（只在 CL 上用时间衰减） --------
        if self.use_cl_ft and self.ssl_lambda > 0:
            u_norm = F.normalize(u_vec, dim=1)
            p_norm = F.normalize(p_vec, dim=1)

            pos_sim = torch.sum(u_norm * p_norm, dim=1)
            pos_exp = torch.exp(pos_sim / self.tau)

            sim_matrix = torch.matmul(u_norm, p_norm.t())
            mask = torch.eye(batch_size, device=self.device).bool()
            sim_matrix = sim_matrix.masked_fill(mask, -1e9)

            neg_exp_sum = torch.sum(torch.exp(sim_matrix / self.tau), dim=1)
            cl_per_sample = -torch.log(pos_exp / (neg_exp_sum + 1e-12))

            # 只给 CL 用时间权重
            time_weight = self.compute_time_weight(pos_times)
            if (time_weight is None) or (time_weight.shape[0] != cl_per_sample.shape[0]):
                cl_weight = torch.ones_like(cl_per_sample, device=self.device)
            else:
                cl_weight = time_weight

            cl_loss = (cl_weight * cl_per_sample).mean() * self.ssl_lambda
        else:
            cl_loss = torch.tensor(0.0, device=self.device)

        # -------- PISA-style 稳定对齐：λ_su * ||u_t - u_{t-1}||^2 --------
        if (self.align_lambda > 0) and (self.prev_user_emb is not None) and (self.user_align_weight is not None):
            prev_u = self.prev_user_emb[users]
            weight = self.user_align_weight[users]
            diff = u_vec - prev_u
            per_user_align = torch.sum(diff * diff, dim=1)
            align_loss = (weight * per_user_align).mean() * self.align_lambda
        else:
            align_loss = torch.tensor(0.0, device=self.device)

        return bpr_loss, reg_loss, cl_loss, align_loss

    # ---------- Trainer 接口 ----------
    def cal_loss(self, batch_data):
        if len(batch_data) >= 4:
            users, pos_items, neg_items, pos_times = batch_data
        elif len(batch_data) == 3:
            users, pos_items, neg_items = batch_data
            pos_times = None
        else:
            raise ValueError("batch_data 至少需要 (users, pos_items, neg_items) 三个元素")

        users = users.to(self.device)
        pos_items = pos_items.to(self.device)
        neg_items = neg_items.to(self.device)
        if pos_times is not None:
            pos_times = pos_times.to(self.device)

        bpr_loss, reg_loss, cl_loss, align_loss = self.forward(users, pos_items, neg_items, pos_times)
        total_loss = bpr_loss + reg_loss + cl_loss + align_loss

        loss_dict = {
            "bpr_loss": float(bpr_loss.item()),
            "reg_loss": float(reg_loss.item()),
            "na_loss": float(cl_loss.item()),
            "align_loss": float(align_loss.item()),
        }
        return total_loss, loss_dict

    # ---------- 导出 embedding ----------
    @torch.no_grad()
    def generate(self):
        user_gate = torch.sigmoid(self.user_gate_raw)
        item_gate = torch.sigmoid(self.item_gate_raw)

        u_base = self.base_user_emb.weight.detach()
        i_base = self.base_item_emb.weight.detach()
        u_delta = self.delta_user_emb.weight.detach()
        i_delta = self.delta_item_emb.weight.detach()

        user_emb = u_base + user_gate * u_delta
        item_emb = i_base + item_gate * i_delta
        return user_emb.to(self.device), item_emb.to(self.device)

    # ---------- rating ----------
    @torch.no_grad()
    def rating(self, user_input, item_emb=None):
        if item_emb is None:
            user_ids = user_input.long().to(self.device)
            user_emb_all, item_emb_all = self.generate()
            u = user_emb_all[user_ids]
            scores = torch.matmul(u, item_emb_all.t())
            return scores
        else:
            u = user_input.to(self.device)
            i = item_emb.to(self.device)
            return torch.matmul(u, i.t())
