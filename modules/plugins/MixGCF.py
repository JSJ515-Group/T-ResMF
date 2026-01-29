import torch
import torch.nn as nn
from modules.plugins.GraphProPluginModel import GraphProPluginModel
from utils.parse_args import args
import torch.nn.functional as F
from modules.utils import EdgelistDrop
import logging
from modules.utils import scatter_add, scatter_sum

init = nn.init.xavier_uniform_
logger = logging.getLogger('train_logger')


class MixGCF(GraphProPluginModel):
    def __init__(self, dataset, pretrained_model=None, phase='pretrain'):
        super().__init__(dataset, pretrained_model, phase)

        # 和 GraphPro 一样构建双向归一化邻接
        self.adj = self._make_binorm_adj(dataset.graph)
        self.edges = self.adj._indices().t()
        self.edge_norm = self.adj._values()

        # 负样本个数：优先用 args.n_negs，没有就回退到 args.neg_num，最后默认 1
        self.n_negs = getattr(args, "n_negs", getattr(args, "neg_num", 1))

        self.edge_dropout = EdgelistDrop()

    def _agg(self, all_emb, edges, edge_norm):
        src_emb = all_emb[edges[:, 0]]

        # bi-norm
        src_emb = src_emb * edge_norm.unsqueeze(1)

        # conv
        dst_emb = scatter_sum(src_emb, edges[:, 1],
                              dim=0,
                              dim_size=self.num_users + self.num_items)
        return dst_emb

    def _edge_binorm(self, edges):
        user_degs = scatter_add(torch.ones_like(edges[:, 0]),
                                edges[:, 0],
                                dim=0,
                                dim_size=self.num_users)
        user_degs = user_degs[edges[:, 0]]

        item_degs = scatter_add(torch.ones_like(edges[:, 1]),
                                edges[:, 1],
                                dim=0,
                                dim_size=self.num_items)
        item_degs = item_degs[edges[:, 1]]

        norm = torch.pow(user_degs, -0.5) * torch.pow(item_degs, -0.5)
        return norm

    def forward(self, edges, edge_norm, edge_times=None, return_res_emb=False):
        """
        前向传播：
        - 融合时间编码：edge_norm = 0.5 * (结构权重 + 时间权重)
        - emb_gate 用的还是 GraphPro 原来的门控
        - 输出为最后一层残差求和的 user/item embedding
        """
        if self.phase not in ['vanilla']:
            time_norm = self._relative_edge_time_encoding(edges, edge_times)
            edge_norm = 0.5 * edge_norm + 0.5 * time_norm

        all_emb = torch.cat([self.user_embedding, self.item_embedding], dim=0)
        all_emb = self.emb_gate(all_emb)

        res_emb = [all_emb]
        for _ in range(args.num_layers):
            all_emb = self._agg(res_emb[-1], edges, edge_norm)
            res_emb.append(all_emb)

        res_sum = sum(res_emb)
        user_res_emb, item_res_emb = res_sum.split(
            [self.num_users, self.num_items], dim=0
        )

        if return_res_emb:
            # 注意返回的是：最终 user/item embedding + 每一层的堆叠列表
            return user_res_emb, item_res_emb, res_emb
        return user_res_emb, item_res_emb

    def negative_sampling(self, user_gcn_emb, item_gcn_emb,
                          user, neg_candidates, pos_item):
        """
        MixGCF 原始的 “正样本 + 负样本混合 + hop 选择” 逻辑。
        输入：
            user_gcn_emb: (U, L+1, D)
            item_gcn_emb: (I, L+1, D)
            user:         (B,)
            neg_candidates: (B, n_negs)
            pos_item:       (B,)
        输出：
            neg_emb: (B, L+1, D)   # 每个 hop 选出的负样本 embedding
        """
        batch_size = user.shape[0]

        # s_e, p_e: (B, L+1, D)
        s_e = user_gcn_emb[user]
        p_e = item_gcn_emb[pos_item]

        B, H, D = s_e.shape  # H = L+1

        # 正样本 mixing 的随机系数 (0, 1)
        seed = torch.rand(B, 1, H, 1, device=p_e.device)

        # neg item embedding: (B, n_negs, H, D)
        n_e = item_gcn_emb[neg_candidates].view(
            batch_size, self.n_negs, H, D
        )

        # positive mixing: (B, n_negs, H, D)
        n_e_ = seed * p_e.unsqueeze(1) + (1.0 - seed) * n_e

        # hop mixing 打分: (B, n_negs, H)
        scores = (s_e.unsqueeze(1) * n_e_).sum(dim=-1)

        # 对每个 (batch, hop) 选一个最优负样本索引: indices (B, H)
        indices = torch.max(scores, dim=1)[1].detach()  # argmax over n_negs

        # 把 n_e_ 变成 (B, H, n_negs, D)，方便沿 n_negs 维度 gather
        n_e_trans = n_e_.permute(0, 2, 1, 3)  # (B, H, n_negs, D)

        # === 关键修改：index 也要是 4 维，和 input 维度一致 ===
        # indices: (B, H) -> (B, H, 1, D)
        idx = indices.unsqueeze(-1).unsqueeze(-1).expand(B, H, 1, D)
        # 在 dim=2 上 gather，输出 (B, H, 1, D)，再 squeeze 掉 dim=2 -> (B, H, D)
        neg_emb = n_e_trans.gather(2, idx).squeeze(2)

        return neg_emb

    def cal_loss(self, batch_data):
        """
        训练时的损失：
        - 边 dropout 使用 GraphPro 一样的 EdgelistDrop
        - 支持 batch_data 为 (u, pos, neg) 或 (u, pos, neg, extra...) 两种情况
        """
        # edge dropout：传入 keep_rate = 1 - edge_dropout
        edges, dropout_mask = self.edge_dropout(
            self.edges,
            1 - args.edge_dropout,
            return_mask=True
        )
        edge_norm = self.edge_norm[dropout_mask]

        if self.phase not in ['vanilla']:
            edge_times = self.edge_times[dropout_mask]
        else:
            edge_times = None

        # --- 兼容 3 或 4 个返回值的 DataLoader ---
        if len(batch_data) >= 3:
            users, pos_items, neg_items = batch_data[:3]
        else:
            raise ValueError(
                f"batch_data 长度为 {len(batch_data)}，但 MixGCF 期望至少包含 "
                f"(users, pos_items, neg_items)"
            )

        # forward：拿到最终 embedding + 每层 embedding 列表
        user_emb, item_emb, res_emb = self.forward(
            edges, edge_norm, edge_times=edge_times, return_res_emb=True
        )

        # res_emb: list 长度 L+1，每个是 (U+I, D)
        # -> stack: (U+I, L+1, D)，再拆成 user/item
        stack_emb = torch.stack(res_emb, dim=1)  # (U+I, L+1, D)
        user_stack_emb, item_stack_emb = stack_emb.split(
            [self.num_users, self.num_items], dim=0
        )  # (U, L+1, D), (I, L+1, D)

        # 通过 MixGCF 的负样本混合得到 (B, L+1, D)，再对 hop 求和 -> (B, D)
        neg_item_emb = self.negative_sampling(
            user_stack_emb, item_stack_emb, users, neg_items, pos_items
        ).sum(dim=1)  # (B, D)

        batch_user_emb = user_emb[users]       # (B, D)
        pos_item_emb = item_emb[pos_items]     # (B, D)

        # BPR 推荐损失 + L2 正则
        rec_loss = self._bpr_loss(batch_user_emb, pos_item_emb, neg_item_emb)
        reg_loss = args.weight_decay * self._reg_loss(users, pos_items, neg_items)

        loss = rec_loss + reg_loss
        loss_dict = {
            "rec_loss": rec_loss.item(),
            "reg_loss": reg_loss.item(),
        }
        return loss, loss_dict

    @torch.no_grad()
    def generate(self):
        # 用完整图 + 时间编码生成最终 user/item embedding
        return self.forward(self.edges,
                            self.edge_norm,
                            edge_times=self.edge_times)

    @torch.no_grad()
    def rating(self, user_emb, item_emb):
        return torch.matmul(user_emb, item_emb.t())

    def _reg_loss(self, users, pos_items, neg_items):
        """
        L2 正则，这里 neg_items 可能是 (B, n_negs)，直接按整体 norm 即可。
        """
        u_emb = self.user_embedding[users]          # (B, D)
        pos_i_emb = self.item_embedding[pos_items]  # (B, D)
        neg_i_emb = self.item_embedding[neg_items]  # (B, n_negs, D) 或 (B, D)

        reg_loss = (0.5 * (
                u_emb.norm(2).pow(2) +
                pos_i_emb.norm(2).pow(2) +
                neg_i_emb.norm(2).pow(2)
        ) / float(len(users)))
        return reg_loss

    def forward_lgn(self, edges, edge_norm, edge_times=None, return_layers=False):
        """
        纯 LightGCN 式前向（不加时间），主要是和原代码保持接口一致，
        一般 generate_lgn / forward_lgn 不会在插件模式下用到。
        """
        all_emb = torch.cat([self.user_embedding, self.item_embedding], dim=0)
        res_emb = [all_emb]
        for _ in range(args.num_layers):
            all_emb = self._agg(res_emb[-1], edges, edge_norm)
            res_emb.append(all_emb)

        if not return_layers:
            res_sum = sum(res_emb)
            user_res_emb, item_res_emb = res_sum.split(
                [self.num_users, self.num_items], dim=0
            )
        else:
            user_res_emb, item_res_emb = [], []
            for emb in res_emb:
                u_emb, i_emb = emb.split(
                    [self.num_users, self.num_items], dim=0
                )
                user_res_emb.append(u_emb)
                item_res_emb.append(i_emb)
        return user_res_emb, item_res_emb

    @torch.no_grad()
    def generate_lgn(self, return_layers=False):
        return self.forward_lgn(self.edges,
                                self.edge_norm,
                                return_layers=return_layers)
