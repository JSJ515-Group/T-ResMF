import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.plugins.GraphProPluginModel import GraphProPluginModel
from utils.parse_args import args
from modules.utils import EdgelistDrop
from modules.utils import scatter_add, scatter_sum

init = nn.init.xavier_uniform_


class CGCL(GraphProPluginModel):
    """
    GraphPro 风格 CGCL（适配你的 GraphProPluginModel 框架）：
    - forward(edges, edge_norm, edge_times) -> user_emb, item_emb, embeddings_list
    - SSL: layer_loss / cand_loss / struct_loss（对齐你贴的 RecBole CGCL 实现）
    - time: edge_norm <- 0.5*edge_norm + 0.5*time_norm （phase != vanilla）
    """

    def __init__(self, dataset, pretrained_model=None, phase='pretrain'):
        super().__init__(dataset, pretrained_model, phase)

        # binorm adj
        self.adj = self._make_binorm_adj(dataset.graph)
        self.edges = self.adj._indices().t()    # [E, 2]
        self.edge_norm = self.adj._values()     # [E]

        # ===== CGCL 超参（从 args 中取，没给就用默认）=====
        self.ssl_temp = getattr(args, "ssl_temp", getattr(args, "temp", 0.2))

        self.alpha = getattr(args, "alpha", 0.5)
        self.beta = getattr(args, "beta", 0.5)
        self.gamma = getattr(args, "gamma", 0.5)

        self.ssl_reg_alpha = getattr(args, "ssl_reg_alpha", 1.0)
        self.ssl_reg_beta  = getattr(args, "ssl_reg_beta", 1.0)
        self.ssl_reg_gamma = getattr(args, "ssl_reg_gamma", 1.0)

        # edge dropout（可设为 0 以贴近原 CGCL：不做图增强）
        self.edge_dropout = EdgelistDrop()
        self.keep_rate = 1 - getattr(args, "edge_dropout", 0.0)  # args.edge_dropout=0 -> keep_rate=1

    # ---------- Graph propagation ----------
    def _agg(self, all_emb, edges, edge_norm):
        # all_emb: [N, D], edges: [E,2], edge_norm: [E]
        src_emb = all_emb[edges[:, 0]]                    # [E, D]
        src_emb = src_emb * edge_norm.unsqueeze(1)        # bi-norm weight
        dst_emb = scatter_sum(
            src_emb,
            edges[:, 1],
            dim=0,
            dim_size=self.num_users + self.num_items
        )
        return dst_emb

    def forward(self, edges, edge_norm, edge_times=None):
        # GraphPro 时间注入（非 vanilla）
        if self.phase not in ['vanilla']:
            # 你的 GraphProPluginModel 在 non-vanilla 下会提供 edge_times
            time_norm = self._relative_edge_time_encoding(edges, edge_times)  # [E]
            edge_norm = 0.5 * edge_norm + 0.5 * time_norm

        all_embeddings = torch.cat([self.user_embedding, self.item_embedding], dim=0)  # [N, D]
        all_embeddings = self.emb_gate(all_embeddings)

        embeddings_list = [all_embeddings]  # layer0 = center
        for _ in range(args.num_layers):
            all_embeddings = self._agg(all_embeddings, edges, edge_norm)
            embeddings_list.append(all_embeddings)

        # CGCL 源码里最后是 mean(L+1) 得到最终 embedding
        stack_emb = torch.stack(embeddings_list[:args.num_layers + 1], dim=1)  # [N, L+1, D]
        lightgcn_all = torch.mean(stack_emb, dim=1)                            # [N, D]

        user_all, item_all = lightgcn_all.split([self.num_users, self.num_items], dim=0)
        return user_all, item_all, embeddings_list

    # ---------- SSL loss 1 ----------
    def ssl_layer_loss(self, current_embedding, previous_embedding, users, items):
        # current vs previous (center)
        cur_u_all, cur_i_all = current_embedding.split([self.num_users, self.num_items], dim=0)
        pre_u_all, pre_i_all = previous_embedding.split([self.num_users, self.num_items], dim=0)

        # user tower
        cur_u = cur_u_all[users]
        pre_u = pre_u_all[users]
        norm_u1 = F.normalize(cur_u, dim=1)
        norm_u2 = F.normalize(pre_u, dim=1)
        norm_u_all = F.normalize(pre_u_all, dim=1)

        pos_u = torch.mul(norm_u1, norm_u2).sum(dim=1)
        ttl_u = torch.matmul(norm_u1, norm_u_all.t())
        pos_u = torch.exp(pos_u / self.ssl_temp)
        ttl_u = torch.exp(ttl_u / self.ssl_temp).sum(dim=1)
        loss_u = -torch.log(pos_u / ttl_u).sum()

        # item tower
        cur_i = cur_i_all[items]
        pre_i = pre_i_all[items]
        norm_i1 = F.normalize(cur_i, dim=1)
        norm_i2 = F.normalize(pre_i, dim=1)
        norm_i_all = F.normalize(pre_i_all, dim=1)

        pos_i = torch.mul(norm_i1, norm_i2).sum(dim=1)
        ttl_i = torch.matmul(norm_i1, norm_i_all.t())
        pos_i = torch.exp(pos_i / self.ssl_temp)
        ttl_i = torch.exp(ttl_i / self.ssl_temp).sum(dim=1)
        loss_i = -torch.log(pos_i / ttl_i).sum()

        return self.ssl_reg_alpha * (self.alpha * loss_u + (1.0 - self.alpha) * loss_i)

    # ---------- SSL loss 2 ----------
    def ssl_canditation_layer_loss(self, current_embedding, previous_embedding, users, items):
        # cross-tower (candidate vs center)
        layer_u_all, layer_i_all = current_embedding.split([self.num_users, self.num_items], dim=0)
        pre_u_all, pre_i_all = previous_embedding.split([self.num_users, self.num_items], dim=0)

        # user tower: current_user_embeddings = layer_item_embeddings[item]
        pre_u = pre_u_all[users]
        cur_u = layer_i_all[items]

        norm_u1 = F.normalize(cur_u, dim=1)
        norm_u2 = F.normalize(pre_u, dim=1)
        norm_u_all = F.normalize(pre_u_all, dim=1)

        pos_u = torch.mul(norm_u1, norm_u2).sum(dim=1)
        ttl_u = torch.matmul(norm_u1, norm_u_all.t())
        pos_u = torch.exp(pos_u / self.ssl_temp)
        ttl_u = torch.exp(ttl_u / self.ssl_temp).sum(dim=1)
        loss_u = -torch.log(pos_u / ttl_u).sum()

        # item tower: current_item_embeddings = layer_user_embeddings[user]
        pre_i = pre_i_all[items]
        cur_i = layer_u_all[users]

        norm_i1 = F.normalize(cur_i, dim=1)
        norm_i2 = F.normalize(pre_i, dim=1)
        norm_i_all = F.normalize(pre_i_all, dim=1)

        pos_i = torch.mul(norm_i1, norm_i2).sum(dim=1)
        ttl_i = torch.matmul(norm_i1, norm_i_all.t())
        pos_i = torch.exp(pos_i / self.ssl_temp)
        ttl_i = torch.exp(ttl_i / self.ssl_temp).sum(dim=1)
        loss_i = -torch.log(pos_i / ttl_i).sum()

        return self.ssl_reg_beta * (self.beta * loss_u + (1.0 - self.beta) * loss_i)

    # ---------- SSL loss 3 ----------
    def calcuate_struct_loss(self, neighbor_embedding, center_embedding, users, items):
        neigh_u_all, neigh_i_all = neighbor_embedding.split([self.num_users, self.num_items], dim=0)
        cent_u_all, cent_i_all = center_embedding.split([self.num_users, self.num_items], dim=0)

        # user side: item as anchor vs user center
        cent_u = cent_u_all[users]
        neigh_i = neigh_i_all[items]

        neigh_i = F.normalize(neigh_i, dim=1)
        cent_u = F.normalize(cent_u, dim=1)
        cent_u_all_norm = F.normalize(cent_u_all, dim=1)

        pos_u = torch.mul(neigh_i, cent_u).sum(dim=1)
        ttl_u = torch.matmul(neigh_i, cent_u_all_norm.t())
        pos_u = torch.exp(pos_u / self.ssl_temp)
        ttl_u = torch.exp(ttl_u / self.ssl_temp).sum(dim=1)
        loss_u = -torch.log(pos_u / ttl_u).sum()

        # item side: user as anchor vs item center
        neigh_u = neigh_u_all[users]
        cent_i = cent_i_all[items]

        neigh_u = F.normalize(neigh_u, dim=1)
        cent_i = F.normalize(cent_i, dim=1)
        cent_i_all_norm = F.normalize(cent_i_all, dim=1)

        pos_i = torch.mul(neigh_u, cent_i).sum(dim=1)
        ttl_i = torch.matmul(neigh_u, cent_i_all_norm.t())
        pos_i = torch.exp(pos_i / self.ssl_temp)
        ttl_i = torch.exp(ttl_i / self.ssl_temp).sum(dim=1)
        loss_i = -torch.log(pos_i / ttl_i).sum()

        return self.ssl_reg_gamma * (self.gamma * loss_u + (1.0 - self.gamma) * loss_i)

    # ---------- GraphPro training API ----------
    def cal_loss(self, batch_data):
        # 兼容 GraphPro dataloader (u, pos, neg, extra...)
        users, pos_items, neg_items = batch_data[:3]

        # one graph (optional edge dropout)
        edges, mask = self.edge_dropout(self.edges, self.keep_rate, return_mask=True)
        edge_norm = self.edge_norm[mask]

        if self.phase not in ['vanilla']:
            edge_times = self.edge_times[mask]
        else:
            edge_times = None

        user_all, item_all, emb_list = self.forward(edges, edge_norm, edge_times=edge_times)

        # 对齐你贴的 CGCL：center=layer0, candidate=layer1, context=layer2
        center = emb_list[0]
        candidate = emb_list[1] if len(emb_list) > 1 else emb_list[0]
        context = emb_list[2] if len(emb_list) > 2 else emb_list[-1]

        layer_loss = self.ssl_layer_loss(context, center, users, pos_items) if self.ssl_reg_alpha > 1e-20 else center.new_tensor(0.0)
        can_loss   = self.ssl_canditation_layer_loss(candidate, center, users, pos_items) if self.ssl_reg_beta > 1e-20 else center.new_tensor(0.0)
        str_loss   = self.calcuate_struct_loss(context, candidate, users, pos_items) if self.ssl_reg_gamma > 1e-20 else center.new_tensor(0.0)

        # BPR + reg
        rec_loss = self._bpr_loss(user_all[users], item_all[pos_items], item_all[neg_items])
        reg_loss = getattr(args, "weight_decay", 0.0) * self._reg_loss(users, pos_items, neg_items)

        loss = rec_loss + reg_loss + layer_loss + can_loss + str_loss
        loss_dict = {
            "rec_loss": rec_loss.item(),
            "reg_loss": reg_loss.item(),
            "layer_loss": layer_loss.item(),
            "can_loss": can_loss.item(),
            "str_loss": str_loss.item(),
        }
        return loss, loss_dict

    def _reg_loss(self, users, pos_items, neg_items):
        u = self.user_embedding[users]
        pi = self.item_embedding[pos_items]
        ni = self.item_embedding[neg_items]
        return 0.5 * (u.norm(2).pow(2) + pi.norm(2).pow(2) + ni.norm(2).pow(2)) / float(len(users))

    @torch.no_grad()
    def generate(self):
        user_all, item_all, _ = self.forward(self.edges, self.edge_norm, edge_times=self.edge_times)
        return user_all, item_all

    @torch.no_grad()
    def rating(self, user_emb, item_emb):
        return torch.matmul(user_emb, item_emb.t())
