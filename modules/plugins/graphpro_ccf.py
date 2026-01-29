import torch
from torch import nn
from utility import losses


class GraphProCCF(nn.Module):
    def __init__(self, args, dataset, device):
        super(GraphProCCF, self).__init__()
        self.args = args
        self.dataset = dataset
        self.device = device

        # === 使用 LightCCF 的 embedding 初始化 ===
        self.user_embedding = nn.Embedding(dataset.num_users, args.embedding_size)
        self.item_embedding = nn.Embedding(dataset.num_items, args.embedding_size)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # === 图结构 & prompt ===
        self.adj_mat = dataset.sparse_adjacency_matrix()
        self.adj_mat = self.adj_mat.to(device)
        self.prompt_embedding = nn.Parameter(torch.randn(args.embedding_size))

        self.tau = args.tau
        self.ssl_lambda = args.ssl_lambda
        self.reg_lambda = args.reg_lambda

    def forward(self, user, pos_item, neg_item, adj_mat=None):
        if adj_mat is None:
            adj_mat = self.adj_mat

        # --- LightCCF的聚合 ---
        user_emb_all, item_emb_all = self.aggregate(adj_mat)
        u, pos, neg = user_emb_all[user], item_emb_all[pos_item], item_emb_all[neg_item]

        # --- loss 计算 ---
        bpr_loss = losses.get_bpr_loss(u, pos, neg)
        reg_loss = losses.get_reg_loss(u, pos, neg) * self.reg_lambda
        na_loss = losses.get_neighbor_aggregate_loss(u, pos, self.tau) * self.ssl_lambda

        total_loss = bpr_loss + reg_loss + na_loss
        return total_loss

    def aggregate(self, adj_mat):
        # === LightCCF 原聚合逻辑 ===
        embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [embeddings]
        for _ in range(self.args.gcn_layer):
            embeddings = torch.sparse.mm(adj_mat, embeddings)
            all_embeddings.append(embeddings)
        final_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        user_emb, item_emb = torch.split(final_embeddings, [self.dataset.num_users, self.dataset.num_items])
        return user_emb, item_emb
