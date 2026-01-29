import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.plugins.GraphProPluginModel import GraphProPluginModel
from utils.parse_args import args
from modules.utils import EdgelistDrop
from modules.utils import scatter_add, scatter_sum

init = nn.init.xavier_uniform_


def cal_infonce(view1, view2, temperature, b_cos=True, eps=1e-8):
    if b_cos:
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)                 # [N]
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))  # [N, N]
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    cl_loss = -torch.log(pos_score / (ttl_score + eps))
    return torch.mean(cl_loss)


class VGCL(GraphProPluginModel):
    """
    Clean-room 可跑版 VGCL（不等同作者原版实现）：
    - 两个 view：在“初始 embedding”上做 feature dropout（虚拟视图），而不是 edge dropout
    - 然后用同一张图传播得到 view embedding，做 InfoNCE
    - GraphPro 时间：edge_norm = 0.5*edge_norm + 0.5*time_norm（非 vanilla）
    """
    def __init__(self, dataset, pretrained_model=None, phase='pretrain'):
        super().__init__(dataset, pretrained_model, phase)

        self.adj = self._make_binorm_adj(dataset.graph)
        self.edges = self.adj._indices().t()
        self.edge_norm = self.adj._values()

        self.temp = getattr(args, "temp", 0.2)
        self.lbd = getattr(args, "lbd", getattr(args, "ssl_lambda", 0.1))
        self.feat_drop = getattr(args, "feat_drop", 0.1)   # VGCL 的“虚拟视图”dropout
        self.cl_max_nodes = getattr(args, "cl_max_nodes", 4096)

        self.edge_dropout = EdgelistDrop()

    def _cap_idx(self, idx: torch.Tensor):
        idx = torch.unique(idx).long().to(args.device)
        if idx.numel() > self.cl_max_nodes:
            perm = torch.randperm(idx.numel(), device=idx.device)[:self.cl_max_nodes]
            idx = idx[perm]
        return idx

    def _agg(self, all_emb, edges, edge_norm):
        src_emb = all_emb[edges[:, 0]]
        src_emb = src_emb * edge_norm.unsqueeze(1)
        dst_emb = scatter_sum(src_emb, edges[:, 1], dim=0, dim_size=self.num_users + self.num_items)
        return dst_emb

    def forward(self, edges, edge_norm, edge_times=None, feat_dropout=False):
        # GraphPro 时间注入
        if self.phase not in ['vanilla']:
            time_norm = self._relative_edge_time_encoding(edges, edge_times)
            edge_norm = 0.5 * edge_norm + 0.5 * time_norm

        all_emb = torch.cat([self.user_embedding, self.item_embedding], dim=0)
        all_emb = self.emb_gate(all_emb)

        # VGCL：用 feature dropout 生成“虚拟视图”
        if feat_dropout and self.training:
            all_emb = F.dropout(all_emb, p=self.feat_drop, training=True)

        res_emb = [all_emb]
        for _ in range(args.num_layers):
            all_emb = self._agg(res_emb[-1], edges, edge_norm)
            res_emb.append(all_emb)

        res_emb = sum(res_emb)
        user_res_emb, item_res_emb = res_emb.split([self.num_users, self.num_items], dim=0)
        return user_res_emb, item_res_emb

    def cal_cl_loss(self, idx, edges, edge_norm, edge_times=None):
        users, pos_items = idx[0], idx[1]
        u_idx = self._cap_idx(users)
        i_idx = self._cap_idx(pos_items)

        # 两次 feature-drop view
        u1, it1 = self.forward(edges, edge_norm, edge_times=edge_times, feat_dropout=True)
        u2, it2 = self.forward(edges, edge_norm, edge_times=edge_times, feat_dropout=True)

        user_cl = cal_infonce(u1[u_idx], u2[u_idx], self.temp)
        item_cl = cal_infonce(it1[i_idx], it2[i_idx], self.temp)
        return user_cl + item_cl

    def cal_loss(self, batch_data):
        # 用“完整图”即可（VGCL 的 view 在 feature-drop 上做），这里也可以按你习惯做一次 edge dropout
        keep_rate = 1 - getattr(args, "edge_dropout", 0.5)
        edges, mask = self.edge_dropout(self.edges, keep_rate, return_mask=True)
        edge_norm = self.edge_norm[mask]
        edge_times = self.edge_times[mask] if self.phase not in ['vanilla'] else None

        users, pos_items, neg_items = batch_data[:3]

        user_emb, item_emb = self.forward(edges, edge_norm, edge_times=edge_times, feat_dropout=False)

        rec_loss = self._bpr_loss(user_emb[users], item_emb[pos_items], item_emb[neg_items])
        reg_loss = getattr(args, "weight_decay", 0.0) * self._reg_loss(users, pos_items, neg_items)
        cl_loss = self.lbd * self.cal_cl_loss([users, pos_items], edges, edge_norm, edge_times=edge_times)

        loss = rec_loss + reg_loss + cl_loss
        return loss, {"rec_loss": rec_loss.item(), "reg_loss": reg_loss.item(), "cl_loss": cl_loss.item()}

    @torch.no_grad()
    def generate(self):
        return self.forward(self.edges, self.edge_norm, edge_times=self.edge_times, feat_dropout=False)

    @torch.no_grad()
    def rating(self, user_emb, item_emb):
        return torch.matmul(user_emb, item_emb.t())

    def _reg_loss(self, users, pos_items, neg_items):
        u_emb = self.user_embedding[users]
        pos_i_emb = self.item_embedding[pos_items]
        neg_i_emb = self.item_embedding[neg_items]
        reg_loss = 0.5 * (u_emb.norm(2).pow(2) + pos_i_emb.norm(2).pow(2) + neg_i_emb.norm(2).pow(2)) / float(len(users))
        return reg_loss
