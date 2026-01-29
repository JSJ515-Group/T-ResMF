import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from modules.plugins.GraphProPluginModel import GraphProPluginModel
from utils.parse_args import args
from modules.utils import EdgelistDrop
from modules.utils import scatter_sum

logger = logging.getLogger("train_logger")


def _xavier_normal_init(module: nn.Module):
    """RecDCL 原实现用 xavier_normal 初始化 Linear/Embedding。这里给 projector/predictor 用即可。"""
    if isinstance(module, nn.Embedding):
        nn.init.xavier_normal_(module.weight.data)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight.data)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)


class RecDCL(GraphProPluginModel):
    """
    GraphPro 风格 RecDCL：
    - encoder: MF / LightGCN（默认 LightGCN）
    - loss: BT + Poly + Momentum (与你贴的 RecDCL 代码一致)
    - time: phase != vanilla 时，将 time_norm 融合进 edge_norm
    """

    def __init__(self, dataset, pretrained_model=None, phase="pretrain"):
        super().__init__(dataset, pretrained_model, phase)

        # 图（GraphProPluginModel 里已构建过，这里保持你其他插件的写法，重复也无所谓）
        self.adj = self._make_binorm_adj(dataset.graph)
        self.edges = self.adj._indices().t()
        self.edge_norm = self.adj._values()

        # ---------- RecDCL 超参（从 args 取，没配就给默认） ----------
        self.encoder_name = getattr(args, "encoder", "LightGCN")  # 'MF' or 'LightGCN'
        self.n_layers = getattr(args, "num_layers", 3)

        self.reg_weight = getattr(args, "reg_weight", 0.0)  # 可选（若你要加 BPR/正则）
        self.a = getattr(args, "a", 1.0)
        self.polyc = getattr(args, "polyc", 0.0)
        self.degree = getattr(args, "degree", 2)
        self.poly_coeff = getattr(args, "poly_coeff", 0.0)

        self.bt_coeff = getattr(args, "bt_coeff", 1.0)
        self.all_bt_coeff = getattr(args, "all_bt_coeff", 0.0)

        self.mom_coeff = getattr(args, "mom_coeff", 0.0)
        self.momentum = getattr(args, "momentum", 0.9)

        # 可选：为了和 GraphPro 其它模型评测更一致，允许加一项 BPR（默认关）
        self.rec_coeff = getattr(args, "recdcl_rec_coeff", 0.0)  # 0=完全复刻原 RecDCL（无BPR）

        # 可选 edge dropout（RecDCL 原版没有，默认 keep=1）
        self.edge_dropout = EdgelistDrop()
        self.keep_rate = 1 - getattr(args, "edge_dropout", 0.0)

        # ---------- RecDCL 模块 ----------
        self.bn = nn.BatchNorm1d(self.emb_size, affine=False)

        # projector：与你贴的 RecDCL 一致（E->E->E->E 的 MLP，Linear 无 bias，含 BN+ReLU）
        sizes = [self.emb_size, self.emb_size, self.emb_size, self.emb_size]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # predictor
        self.predictor = nn.Linear(self.emb_size, self.emb_size)

        # momentum target history（用 buffer，不参与梯度）
        self.register_buffer("u_target_his", torch.randn(self.num_users, self.emb_size))
        self.register_buffer("i_target_his", torch.randn(self.num_items, self.emb_size))

        # 初始化 projector / predictor（embedding 参数你们 GraphProPluginModel 已经 init 过了）
        self.projector.apply(_xavier_normal_init)
        self.predictor.apply(_xavier_normal_init)

    # ---------- LightGCN 聚合 ----------
    def _agg(self, all_emb, edges, edge_norm):
        src_emb = all_emb[edges[:, 0]]
        src_emb = src_emb * edge_norm.unsqueeze(1)
        dst_emb = scatter_sum(src_emb, edges[:, 1], dim=0, dim_size=self.num_users + self.num_items)
        return dst_emb

    def _encode_all(self, edges, edge_norm, edge_times=None):
        """
        输出全量 user/item embedding（不更新 momentum target）
        """
        if self.phase not in ["vanilla"]:
            time_norm = self._relative_edge_time_encoding(edges, edge_times)  # [E]
            edge_norm = 0.5 * edge_norm + 0.5 * time_norm

        if self.encoder_name == "MF":
            user_all = self.user_embedding
            item_all = self.item_embedding
            return user_all, item_all

        # LightGCN
        all_emb = torch.cat([self.user_embedding, self.item_embedding], dim=0)
        all_emb = self.emb_gate(all_emb)

        emb_list = [all_emb]
        for _ in range(self.n_layers):
            all_emb = self._agg(all_emb, edges, edge_norm)
            emb_list.append(all_emb)

        # RecDCL 原版 LGCNEncoder：mean over layers
        stack = torch.stack(emb_list, dim=1)           # [N, L+1, D]
        out = torch.mean(stack, dim=1)                 # [N, D]
        user_all, item_all = out.split([self.num_users, self.num_items], dim=0)
        return user_all, item_all

    # ---------- RecDCL losses ----------
    @staticmethod
    def off_diagonal(x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def bt(self, x, y):
        ux = self.projector(x)
        iy = self.projector(y)
        c = self.bn(ux).T @ self.bn(iy)
        c.div_(ux.size(0))
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().div(self.emb_size)
        off_diag = self.off_diagonal(c).pow_(2).sum().div(self.emb_size)
        return on_diag + self.bt_coeff * off_diag

    def poly_feature(self, x):
        ux = self.projector(x)
        xx = self.bn(ux).T @ self.bn(ux)
        poly = (self.a * xx + self.polyc) ** self.degree
        return poly.mean().log()

    @staticmethod
    def loss_fn(p, z):
        return -F.cosine_similarity(p, z.detach(), dim=-1).mean()

    # ---------- GraphPro API ----------
    def cal_loss(self, batch_data):
        """
        batch_data: (users, pos_items, neg_items, extra...)
        RecDCL 原版只用 (user,item) 正样本；neg 在这里默认不使用（除非 rec_coeff>0）
        """
        users = batch_data[0]
        pos_items = batch_data[1]
        neg_items = batch_data[2] if len(batch_data) >= 3 else None

        # 图（可选 edge dropout）
        edges, mask = self.edge_dropout(self.edges, self.keep_rate, return_mask=True)
        edge_norm = self.edge_norm[mask]
        if self.phase not in ["vanilla"]:
            edge_times = self.edge_times[mask]
        else:
            edge_times = None

        # 取 batch embedding
        if self.encoder_name == "MF":
            user_e = self.user_embedding[users]
            item_e = self.item_embedding[pos_items]
        else:
            user_all, item_all = self._encode_all(edges, edge_norm, edge_times=edge_times)
            user_e = user_all[users]
            item_e = item_all[pos_items]

        # momentum targets（严格按你贴的代码逻辑）
        with torch.no_grad():
            u_h = self.u_target_his[users].clone()
            i_h = self.i_target_his[pos_items].clone()

            u_target = u_h * self.momentum + user_e.detach() * (1.0 - self.momentum)
            i_target = i_h * self.momentum + item_e.detach() * (1.0 - self.momentum)

            # 注意：你贴的代码是把 history 直接写成当前 user_e/item_e
            self.u_target_his[users] = user_e.detach()
            self.i_target_his[pos_items] = item_e.detach()

        # normalize + predictor
        user_e_n = F.normalize(user_e, dim=-1)
        item_e_n = F.normalize(item_e, dim=-1)

        user_p = self.predictor(user_e)
        item_p = self.predictor(item_e)

        # BT
        bt_loss = user_e.new_tensor(0.0) if self.all_bt_coeff == 0 else self.bt(user_e_n, item_e_n)
        # Poly
        poly_loss = user_e.new_tensor(0.0) if self.poly_coeff == 0 else (
            self.poly_feature(user_e_n) / 2 + self.poly_feature(item_e_n) / 2
        )
        # Momentum
        mom_loss = user_e.new_tensor(0.0) if self.mom_coeff == 0 else (
            self.loss_fn(user_p, i_target) / 2 + self.loss_fn(item_p, u_target) / 2
        )

        loss = self.all_bt_coeff * bt_loss + self.poly_coeff * poly_loss + self.mom_coeff * mom_loss

        # 可选：加 BPR（为了跟你们 GraphPro 其它模型更可比；默认 rec_coeff=0 不加）
        rec_loss = user_e.new_tensor(0.0)
        reg_loss = user_e.new_tensor(0.0)
        if self.rec_coeff > 0 and neg_items is not None:
            if self.encoder_name == "MF":
                neg_e = self.item_embedding[neg_items]
            else:
                neg_e = item_all[neg_items]
            rec_loss = self._bpr_loss(user_e, item_e, neg_e)
            reg_loss = getattr(args, "weight_decay", 0.0) * self._reg_loss(users, pos_items, neg_items)
            loss = loss + self.rec_coeff * rec_loss + reg_loss

        loss_dict = {
            "bt_loss": float(bt_loss.detach().cpu()),
            "poly_loss": float(poly_loss.detach().cpu()),
            "mom_loss": float(mom_loss.detach().cpu()),
            "rec_loss": float(rec_loss.detach().cpu()),
            "reg_loss": float(reg_loss.detach().cpu()),
        }
        return loss, loss_dict

    def _reg_loss(self, users, pos_items, neg_items):
        u = self.user_embedding[users]
        pi = self.item_embedding[pos_items]
        ni = self.item_embedding[neg_items] if neg_items is not None else 0.0
        if isinstance(ni, float):
            return 0.5 * (u.norm(2).pow(2) + pi.norm(2).pow(2)) / float(len(users))
        return 0.5 * (u.norm(2).pow(2) + pi.norm(2).pow(2) + ni.norm(2).pow(2)) / float(len(users))

    @torch.no_grad()
    def generate(self):
        # 用全图生成 user/item embedding（不更新 momentum target）
        user_all, item_all = self._encode_all(self.edges, self.edge_norm, edge_times=self.edge_times)
        return user_all, item_all

    @torch.no_grad()
    def rating(self, user_emb, item_emb):
        return torch.matmul(user_emb, item_emb.t())
