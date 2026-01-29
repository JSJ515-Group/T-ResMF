# modules/plugins/GraphProMFResFinetune1.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.base_model import BaseModel
from utils.parse_args import args

init = nn.init.xavier_uniform_


class GraphProMFResFinetune(BaseModel):
    """
    微调阶段：针对 MF / GraphProCCF 预训练嵌入的残差式微调模型
    - 预训练：MF + CL（GraphProCCF）
    - 微调：base(冻结) + delta(可学习) + BPR + 可选 CL
    - 不使用置信度矩阵
    """

    def __init__(self, dataset, pretrained_user_emb, pretrained_item_emb, phase="finetune"):
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
        # 默认开启；可以通过 --finetune_use_cl 0 关闭
        self.use_cl_ft = bool(getattr(args, "finetune_use_cl", True))
        # 微调用的 CL 权重（不要太大，否则会压制 BPR）
        self.ssl_lambda = float(getattr(args, "ft_ssl_lambda", 2))
        # 温度系数：复用 tau，或者单独给也行
        self.tau = float(getattr(args, "tau", 0.2))

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

        print("=== GraphProMFResFinetune 初始化（带 CL，无置信度矩阵） ===")
        print(f"用户数: {self.num_users}, 物品数: {self.num_items}")
        print(f"Embedding 维度: {self.embedding_size}")
        print(f"使用时间衰减: {self.use_time_decay}, λ={self.time_lambda}")
        print(f"微调阶段使用CL: {self.use_cl_ft}, ssl_lambda={self.ssl_lambda}, tau={self.tau}")
        print("仅使用 BPR + 残差正则 + （可选）CL 进行微调")
        print("==============================================")

    # ----------------------------------------------------------------------
    # 时间衰减权重（batch 内 min-max 归一化，越新权重越大）
    # ----------------------------------------------------------------------
    def compute_time_weight(self, pos_times):
        """
        简化版时间衰减：
        - 对 batch 内时间做 min-max 归一化
        - 越新的样本权重越大:
            w_t = exp(-lambda * (1 - norm_t)), clamp 到 [0.1, 1.0]
        返回形状 [B]，或者 None（未启用）
        """
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

    # ----------------------------------------------------------------------
    # forward：一批 (u, i+, i-) 的 BPR + 可选 CL
    # ----------------------------------------------------------------------
    def forward(self, users, pos_items, neg_items, pos_times=None):
        users = users.long().to(self.device)
        pos_items = pos_items.long().to(self.device)
        neg_items = neg_items.long().to(self.device)
        if pos_times is not None:
            pos_times = pos_times.to(self.device)

        batch_size = users.size(0)

        # gate ∈ (0,1)
        user_gate = torch.sigmoid(self.user_gate_raw)
        item_gate = torch.sigmoid(self.item_gate_raw)

        # base + gate * delta
        u_base = self.base_user_emb(users)
        p_base = self.base_item_emb(pos_items)
        n_base = self.base_item_emb(neg_items)

        u_delta = self.delta_user_emb(users)
        p_delta = self.delta_item_emb(pos_items)
        n_delta = self.delta_item_emb(neg_items)

        u_vec = u_base + user_gate * u_delta
        p_vec = p_base + item_gate * p_delta
        n_vec = n_base + item_gate * n_delta

        # ---- BPR 打分 ----
        pos_scores = torch.sum(u_vec * p_vec, dim=1)
        neg_scores = torch.sum(u_vec * n_vec, dim=1)
        per_sample_bpr = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-12)  # [B]

        # ---- 时间权重 ----
        time_weight = self.compute_time_weight(pos_times)
        if time_weight is None or time_weight.shape[0] != per_sample_bpr.shape[0]:
            time_weight = torch.ones_like(per_sample_bpr, device=self.device)

        # 加权 BPR
        bpr_loss = (time_weight * per_sample_bpr).mean()

        # ---- 残差正则：只约束 delta ----
        reg_loss = 0.5 * (
            u_delta.norm(2).pow(2) +
            p_delta.norm(2).pow(2) +
            n_delta.norm(2).pow(2)
        ) / users.shape[0] * self.reg_lambda

        # ---- 对比学习 InfoNCE（batch 内 user–pos item） ----
        if self.use_cl_ft and self.ssl_lambda > 0:
            # 归一化后的特征
            u_norm = F.normalize(u_vec, dim=1)   # [B, d]
            p_norm = F.normalize(p_vec, dim=1)   # [B, d]

            # 正样本相似度
            pos_sim = torch.sum(u_norm * p_norm, dim=1)      # [B]
            pos_exp = torch.exp(pos_sim / self.tau)          # [B]

            # 所有 user 对所有正 item 的相似度矩阵（InfoNCE 分母）
            sim_matrix = torch.matmul(u_norm, p_norm.t())    # [B, B]
            mask = torch.eye(batch_size, device=self.device).bool()
            sim_matrix = sim_matrix.masked_fill(mask, -1e9)

            neg_exp_sum = torch.sum(torch.exp(sim_matrix / self.tau), dim=1)  # [B]
            cl_per_sample = -torch.log(pos_exp / (neg_exp_sum + 1e-12))       # [B]

            # CL 同样用时间权重（你也可以换成全 1，看你实验习惯）
            cl_weight = time_weight
            cl_loss = (cl_weight * cl_per_sample).mean() * self.ssl_lambda
        else:
            cl_loss = torch.tensor(0.0, device=self.device)

        return bpr_loss, reg_loss, cl_loss

    # ----------------------------------------------------------------------
    # cal_loss：给 Trainer 用的统一接口
    # ----------------------------------------------------------------------
    def cal_loss(self, batch_data):
        """
        batch_data:
            - 可能是 (u, i+, i-)
            - 也可能是 (u, i+, i-, pos_times)
        """
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

        bpr_loss, reg_loss, cl_loss = self.forward(users, pos_items, neg_items, pos_times)
        total_loss = bpr_loss + reg_loss + cl_loss

        loss_dict = {
            "bpr_loss": float(bpr_loss.item()),
            "reg_loss": float(reg_loss.item()),
            "na_loss": float(cl_loss.item()),   # 复用原 GraphPro 的字段名
        }
        return total_loss, loss_dict

    # ----------------------------------------------------------------------
    # generate：输出最终 embedding（base + gate * delta）
    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    # rating：兼容 “传 id” 和 “传 embedding” 两种模式
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def rating(self, user_input, item_emb=None):
        """
        兼容两种调用方式：
        1) rating(user_ids, None)            # user_ids: LongTensor 索引
        2) rating(user_emb, item_emb)        # 显式传入 user/item embedding 矩阵
        """
        if item_emb is None:
            # 情况 1：传进来的是用户 id
            user_ids = user_input.long().to(self.device)
            user_emb_all, item_emb_all = self.generate()
            u = user_emb_all[user_ids]                 # [B, d]
            scores = torch.matmul(u, item_emb_all.t()) # [B, num_items]
            return scores
        else:
            # 情况 2：传进来的是用户 embedding
            u = user_input.to(self.device)             # [B, d]
            i = item_emb.to(self.device)               # [num_items, d] 或 [subset, d]
            return torch.matmul(u, i.t())
