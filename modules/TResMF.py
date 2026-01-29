import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import random
from copy import deepcopy


class TResMF(nn.Module):
    """
    LightGCN-style implementation with temporal fusion
    - 只添加调试输出，不修改任何逻辑
    """

    class TemporalAwareEmbedding(nn.Module):
        """Temporal-aware embedding with time fusion capability"""

        def __init__(self, num_embeddings, embedding_dim, use_temporal_fusion=False, fusion_alpha=0.1):
            super(TResMF.TemporalAwareEmbedding, self).__init__()
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.use_temporal_fusion = use_temporal_fusion
            self.fusion_alpha = fusion_alpha

            # Base embedding
            self.base_embedding = nn.Embedding(num_embeddings, embedding_dim)
            nn.init.xavier_uniform_(self.base_embedding.weight)

            # Time encoder (only created if needed)
            if self.use_temporal_fusion:
                self.time_encoder = nn.Sequential(
                    nn.Linear(1, 8),  # time scalar -> 8 dim
                    nn.ReLU(),
                    nn.Linear(8, embedding_dim),  # 8 dim -> embedding dim
                    nn.Tanh()  # limit output range [-1,1]
                )
                # Initialize time encoder weights
                for layer in self.time_encoder:
                    if hasattr(layer, 'weight'):
                        nn.init.xavier_uniform_(layer.weight)

        def forward(self, indices, times=None):
            # Get base embedding
            base_emb = self.base_embedding(indices)

            # If temporal fusion is enabled and time information is provided
            if self.use_temporal_fusion and times is not None:
                # Ensure correct time data format
                if times.dim() == 1:
                    times = times.unsqueeze(1)  # (B,) -> (B,1)

                # Normalize time to [0,1] range (per-batch normalization)
                times = times.float()
                if torch.max(times) > torch.min(times):
                    times_norm = (times - torch.min(times)) / (torch.max(times) - torch.min(times) + 1e-8)
                else:
                    times_norm = torch.zeros_like(times)

                # Time encoding
                time_emb = self.time_encoder(times_norm)

                # Gentle fusion: base_emb + alpha * time_emb
                fused_emb = base_emb + self.fusion_alpha * time_emb

                return fused_emb
            else:
                # Fallback to base embedding
                return base_emb

        def extra_repr(self):
            return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, " \
                   f"use_temporal_fusion={self.use_temporal_fusion}, fusion_alpha={self.fusion_alpha}"

    def __init__(self, dataset, args, phase=None):
        super(TResMF, self).__init__()
        self.dataset = dataset
        self.args = args
        self.phase = phase
        self.device = getattr(args, 'device', 'cpu')

        # dims / hyperparams
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.embedding_size = int(getattr(args, 'embedding_size', getattr(args, 'emb_size', 64)))
        self.reg_lambda = float(getattr(args, 'reg_lambda', 0.0001))
        self.ssl_lambda = float(getattr(args, 'ssl_lambda', 1.0))
        self.tau = float(getattr(args, 'tau', 0.28))
        self.encoder = getattr(args, 'encoder', 'MF')
        self.gcn_layer = int(getattr(args, 'gcn_layer', getattr(args, 'num_layers', 3)))
        self.use_ccf = bool(getattr(args, 'use_ccf', True))
        self.use_time_decay = bool(getattr(args, 'use_time_decay', False))
        self.time_lambda = float(getattr(args, 'time_decay_lambda', 0.0))

        # New: temporal fusion parameters (only affect embedding representation)
        self.use_temporal_fusion = bool(getattr(args, 'use_temporal_fusion', False))
        self.temporal_fusion_alpha = float(getattr(args, 'temporal_fusion_alpha', 0.1))

        # === New: Fuzzy 置信度加权相关超参 ===
        # 是否启用置信度+余弦相似度加权（默认 True）
        self.use_fuzzy = bool(getattr(args, 'use_fuzzy', True))
        # alpha, beta 控制置信度和相似度的贡献
        self.alpha = float(getattr(args, 'alpha', 0.3))
        self.beta = float(getattr(args, 'beta', 0.7))
        # 预加载置信度矩阵 (user,item) -> confidence
        self.confidence_dict = self._load_precomputed_confidence_dict()

        # edge dropout default
        self.edge_dropout_rate = float(getattr(args, 'edge_dropout', 0.0))
        self.debug_counter = 0
        # 添加调试间隔参数
        self.debug_interval = int(getattr(args, 'debug_interval', 100))

        # embeddings - now using temporal-aware embeddings
        # IMPORTANT: names unchanged, only embedding implementation replaced
        self.user_embedding = self.TemporalAwareEmbedding(
            self.num_users, self.embedding_size,
            use_temporal_fusion=self.use_temporal_fusion,
            fusion_alpha=self.temporal_fusion_alpha
        )
        self.item_embedding = self.TemporalAwareEmbedding(
            self.num_items, self.embedding_size,
            use_temporal_fusion=self.use_temporal_fusion,
            fusion_alpha=self.temporal_fusion_alpha
        )

        # 预计算/缓存邻接矩阵稀疏张量
        self.adj_mat = None
        self.edges = None
        self.edge_norm = None
        if hasattr(dataset, 'sparse_adjacency_matrix'):
            sp_mat = dataset.sparse_adjacency_matrix()
            try:
                adj_norm = self._build_bi_norm_adj(sp_mat)
                self.adj_mat = self._convert_sp_mat_to_sp_tensor(adj_norm).to(self.device)

                coo = adj_norm.tocoo()
                rows = coo.row.astype(np.int64)
                cols = coo.col.astype(np.int64)
                vals = coo.data.astype(np.float32)
                if rows.size > 0:
                    edges_np = np.vstack([rows, cols]).T
                    self.edges = torch.LongTensor(edges_np).to(self.device)
                    self.edge_norm = torch.from_numpy(vals).float().to(self.device)
                else:
                    self.edges = torch.zeros((0, 2), dtype=torch.long, device=self.device)
                    self.edge_norm = torch.zeros((0,), dtype=torch.float, device=self.device)
            except Exception:
                self.adj_mat = None
                self.edges = None
                self.edge_norm = None
        else:
            if hasattr(dataset, 'edgelist_np') and dataset.edgelist_np is not None:
                ed_np = np.array(dataset.edgelist_np, dtype=np.int64)
                if ed_np.size == 0:
                    self.edges = torch.zeros((0, 2), dtype=torch.long, device=self.device)
                    self.edge_norm = torch.zeros((0,), dtype=torch.float, device=self.device)
                else:
                    u = ed_np[:, 0].astype(np.int64)
                    i = ed_np[:, 1].astype(np.int64) + self.num_users
                    rows = np.concatenate([u, i])
                    cols = np.concatenate([i, u])
                    coo = sp.coo_matrix((np.ones_like(rows, dtype=np.float32), (rows, cols)),
                                        shape=(self.num_users + self.num_items, self.num_users + self.num_items))
                    adj_norm = self._build_bi_norm_adj(coo)
                    coo2 = adj_norm.tocoo()
                    rows2 = coo2.row.astype(np.int64)
                    cols2 = coo2.col.astype(np.int64)
                    vals2 = coo2.data.astype(np.float32)
                    edges_np = np.vstack([rows2, cols2]).T
                    self.edges = torch.LongTensor(edges_np).to(self.device)
                    self.edge_norm = torch.from_numpy(vals2).float().to(self.device)
            else:
                self.edges = None
                self.edge_norm = None

        # convenience: train_user_dict if provided
        if hasattr(dataset, 'train_user_dict'):
            self.train_user_dict = dataset.train_user_dict
        else:
            self.train_user_dict = {u: [] for u in range(self.num_users)}
            if hasattr(dataset, 'edgelist_np') and dataset.edgelist_np is not None:
                for u, i in dataset.edgelist_np:
                    self.train_user_dict[int(u)].append(int(i))

        # 添加初始化调试信息
        print("=== GraphProCCF 模型初始化 ===")
        print(f"数据集: {getattr(dataset, 'name', 'unknown')}")
        print(f"用户数: {self.num_users}, 物品数: {self.num_items}")
        print(f"编码器: {self.encoder}, GCN层数: {self.gcn_layer}")
        print(f"使用对比学习: {self.use_ccf}, SSL权重: {self.ssl_lambda}")
        print(f"使用时间衰减: {self.use_time_decay}, 时间λ: {self.time_lambda}")
        print(f"使用时间融合: {self.use_temporal_fusion}, 融合α: {self.temporal_fusion_alpha}")
        print(f"使用Fuzzy置信度加权: {self.use_fuzzy}")
        print(f"调试间隔: 每 {self.debug_interval} 步输出一次")
        print("=" * 50)

    # ======== New: 置信度矩阵加载（用 data_path） ========
    # ======== New: 置信度矩阵加载（Taobao 专用简化版） ========
    def _load_precomputed_confidence_dict(self):
        """
        直接从 ./dataset/taobao/confidence_matrix_3layer.npz 加载置信度矩阵，
        不再依赖 args.dataset_path / args.dataset，避免参数没加导致加载失败。
        """
        # 1) 根目录 & 数据集名先直接硬编码（你现在就是在跑 taobao）
        dataset_root = "./dataset"
        dataset_name = "taobao"

        confidence_file = os.path.join(dataset_root, dataset_name, 'confidence_matrix_3layer.npz')

        if not os.path.exists(confidence_file):
            print()
            return None

        try:
            confidence_sparse = sp.load_npz(confidence_file)  # shape: (num_users, num_items)
            coo = confidence_sparse.tocoo()

            confidence_dict = {}
            for u, i, v in zip(coo.row, coo.col, coo.data):
                # 注意：这里的 (u, i) 使用的是 zhixin.py 映射后的 id，
                # 必须和 GraphPro 读 pretrain.txt 的方式一致（现在我们已经对齐了）
                confidence_dict[(int(u), int(i))] = float(v)

            vals = list(confidence_dict.values())

            if len(vals) > 0:
                print()
            return confidence_dict
        except Exception as e:
            print()
            return None

    def _convert_sp_mat_to_sp_tensor(self, sp_mat):
        """Convert scipy.sparse.coo_matrix to torch.sparse.FloatTensor"""
        coo = sp_mat.tocoo().astype(np.float32)
        if coo.data.size == 0:
            indices = torch.empty((2, 0), dtype=torch.int64)
            values = torch.empty((0,), dtype=torch.float32)
            shape = coo.shape
            tensor = torch.sparse.FloatTensor(indices, values, torch.Size(shape))
            return tensor.coalesce()
        indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
        values = torch.from_numpy(coo.data.astype(np.float32))
        shape = coo.shape
        tensor = torch.sparse.FloatTensor(indices, values, torch.Size(shape))
        return tensor.coalesce()

    def _build_bi_norm_adj(self, sp_mat):
        """Build D^{-1/2} A D^{-1/2}"""
        if not sp.isspmatrix_coo(sp_mat):
            sp_mat = sp_mat.tocoo()
        row_sum = np.array(sp_mat.sum(axis=1)).flatten()
        d_inv_sqrt = np.zeros_like(row_sum, dtype=np.float64)
        nonzero = row_sum > 0
        d_inv_sqrt[nonzero] = np.power(row_sum[nonzero], -0.5)
        rows = sp_mat.row
        cols = sp_mat.col
        vals = sp_mat.data
        vals_norm = vals * d_inv_sqrt[rows] * d_inv_sqrt[cols]
        adj_norm = sp.coo_matrix((vals_norm, (rows, cols)), shape=sp_mat.shape)
        return adj_norm

    def _edge_dropout(self, edges, edge_norm, dropout_rate):
        """Simple edge dropout"""
        if edges is None or edge_norm is None:
            return edges, edge_norm
        if dropout_rate <= 0.0 or not self.training:
            return edges, edge_norm
        E = edges.size(0)
        mask = (torch.rand(E, device=edges.device) >= dropout_rate)
        if mask.sum().item() == 0:
            idx = torch.randint(0, E, (1,), device=edges.device)
            mask[idx] = True
        edges_d = edges[mask]
        edge_norm_d = edge_norm[mask]
        return edges_d, edge_norm_d

    def compute_relative_time_weights(self, pos_times, current_time=None):
        """
        Compute relative time weights with intelligent strategy selection
        Fixes the issue where time range is too small (1-5)
        """
        if pos_times is None:
            if self.training and self.debug_counter % self.debug_interval == 0 and self.use_time_decay:
                print("⚠️ 时间衰减: pos_times为None，返回None")
            return None

        times_flat = pos_times.view(-1).float()

        # Basic statistics
        time_min = times_flat.min().item()
        time_max = times_flat.max().item()
        time_range = time_max - time_min
        unique_count = len(torch.unique(times_flat))

        # 添加调试输出
        if self.training and self.debug_counter % self.debug_interval == 0 and self.use_time_decay:
            print(f"\n=== 时间衰减计算 (step={self.debug_counter}) ===")
            print(f"时间数据统计: min={time_min:.6f}, max={time_max:.6f}, range={time_range:.6f}")
            print(f"唯一值数量: {unique_count}")

        # Strategy selection based on time data characteristics
        if unique_count <= 1 or time_range < 1e-6:
            # All times are the same -> use uniform weights
            if self.training and self.debug_counter % self.debug_interval == 0 and self.use_time_decay:
                print("=== 时间衰减策略: 均匀权重 (所有时间相同) ===")
            return torch.ones_like(times_flat)
        elif time_range < 10:  # Small time range but with diversity
            # Use quantile-based weighting for small ranges
            if self.training and self.debug_counter % self.debug_interval == 0 and self.use_time_decay:
                print(f"=== 时间衰减策略: 分位数加权 (时间范围小但有差异) ===")
            return self.quantile_based_weighting(times_flat)
        else:
            # Normal time data -> standard relative time decay
            if self.training and self.debug_counter % self.debug_interval == 0 and self.use_time_decay:
                print(f"=== 时间衰减策略: 标准相对时间衰减 ===")
            return self.standard_relative_weighting(times_flat, current_time)

    def quantile_based_weighting(self, times):
        """
        Quantile-based weighting for small time ranges
        Uses time ordering rather than absolute differences
        """
        # Sort times and get quantile positions
        sorted_times, indices = torch.sort(times)
        ranks = torch.arange(len(times), device=times.device).float()

        # Normalize ranks to [0,1]
        if len(times) > 1:
            normalized_ranks = ranks / (len(times) - 1)
        else:
            normalized_ranks = torch.ones_like(ranks)

        # Restore original order
        _, reverse_indices = torch.sort(indices)
        quantiles = normalized_ranks[reverse_indices]

        # Apply decay: recent samples (higher quantile) get higher weight
        # Invert: 1 - quantile so recent samples (quantile near 1) get weight near 1
        time_weight = torch.exp(-self.time_lambda * (1.0 - quantiles))

        # Clamp to reasonable range
        time_weight = torch.clamp(time_weight, min=0.1, max=1.0)

        # 添加调试输出
        if self.training and self.debug_counter % self.debug_interval == 0 and self.use_time_decay:
            print(f"分位数加权调试:")
            print(f"分位数范围: {quantiles.min().item():.3f} - {quantiles.max().item():.3f}")
            print(f"时间权重范围: {time_weight.min().item():.3f} - {time_weight.max().item():.3f}")
            print(f"λ参数: {self.time_lambda}")

            # 在 quantile_based_weighting 里，替换原来的打印代码
            old_idx = torch.argmin(quantiles)  # 最旧样本
            new_idx = torch.argmax(quantiles)  # 最新样本



        return time_weight

    def standard_relative_weighting(self, times, current_time=None):
        """基于数据集名称的智能时间单位检测"""
        if current_time is None:
            current_time = times.max()

        time_delta_raw = current_time - times

        # 获取数据集名称
        dataset_name = getattr(self.dataset, 'name', '').lower()

        # 已知数据集的时间单位映射
        dataset_time_units = {
            'amazon': 'days',  # Amazon数据单位是天
            'taobao': 'seconds',  # Taobao数据单位是秒
            'koubei': 'seconds',  # Koubei数据单位是秒
            'yelp': 'days',  # Yelp数据单位是天
            'movielens': 'days',  # MovieLens数据单位是天
        }

        # 优先使用已知数据集的映射
        if dataset_name in dataset_time_units:
            unit = dataset_time_units[dataset_name]
            if unit == 'days':
                conversion_factor = 1.0
                detected_unit = f"天（已知数据集: {dataset_name}）"
            else:  # seconds
                conversion_factor = 24 * 3600
                detected_unit = f"秒（已知数据集: {dataset_name}）"
        else:
            # 回退到自动检测
            detected_unit, conversion_factor = self.detect_time_unit(times)

        time_delta_days = time_delta_raw / conversion_factor
        time_weight = torch.exp(-self.time_lambda * time_delta_days)
        time_weight = torch.clamp(time_weight, min=0.1, max=1.0)

        # 调试输出
        if self.training and self.debug_counter % self.debug_interval == 0 and self.use_time_decay:
            print(f"数据集名称: {dataset_name}")
            print(f"使用时间单位: {detected_unit}")
            print(f"时间差: {time_delta_days.min().item():.6f} - {time_delta_days.max().item():.6f} 天")
            print(f"时间权重: {time_weight.min().item():.3f} - {time_weight.max().item():.3f}")

        return time_weight

    def detect_time_unit(self, times):
        """时间单位检测方法"""
        time_max = times.max().item()
        time_min = times.min().item()
        time_range = time_max - time_min

        if time_range < 1e-6:
            return "unknown", 1.0

        # 基于典型范围检测单位
        if time_range > 10 * 365 * 24 * 3600:  # 10年以上 -> 毫秒
            return "milliseconds", 1000.0 * 24 * 3600
        elif time_range > 5 * 365 * 24 * 3600:  # 5年以上 -> 秒
            return "seconds", 24 * 3600
        elif time_range > 2 * 365:  # 2年以上 -> 天
            return "days", 1.0
        elif time_range > 30:  # 30天以上 -> 天
            return "days", 1.0
        elif time_range > 1:  # 1天以上 -> 天
            return "days", 1.0
        else:  # 小数值
            return "normalized", 1.0

    def aggregate(self, edges=None, edge_norm=None, edge_dropout=0.0):
        """LightGCN aggregation"""
        if self.encoder == 'MF' or (self.adj_mat is None and self.edges is None):
            # return base embeddings (without temporal fusion) for aggregation / evaluation
            user_emb = self.user_embedding.base_embedding.weight.to(self.device)
            item_emb = self.item_embedding.base_embedding.weight.to(self.device)
            return user_emb, item_emb

        device = self.device
        all_emb = torch.cat([self.user_embedding.base_embedding.weight,
                             self.item_embedding.base_embedding.weight], dim=0).to(device)
        embs = [all_emb]

        if edges is None or edge_norm is None:
            if self.edges is None or self.edge_norm is None:
                x = all_emb
                for _ in range(self.gcn_layer):
                    x = torch.sparse.mm(self.adj_mat, x)
                    embs.append(x)
                final_emb = sum(embs)
                user_emb, item_emb = torch.split(final_emb, [self.num_users, self.num_items], dim=0)
                return user_emb, item_emb
            else:
                edges = self.edges
                edge_norm = self.edge_norm

        edges_d, edge_norm_d = self._edge_dropout(edges, edge_norm, edge_dropout)

        x = all_emb
        for _ in range(self.gcn_layer):
            if edges_d.size(0) == 0:
                dst = torch.zeros_like(x)
            else:
                src_emb = x[edges_d[:, 0]]
                src_emb = src_emb * edge_norm_d.unsqueeze(1)
                dst = torch.zeros_like(x)
                dst.index_add_(0, edges_d[:, 1], src_emb)
            x = dst
            embs.append(x)

        final_emb = sum(embs)
        user_emb, item_emb = torch.split(final_emb, [self.num_users, self.num_items], dim=0)
        return user_emb, item_emb

    def forward(self, user, positive, negative, pos_times=None, edge_dropout=0.0):
        """Forward pass with temporal fusion support"""

        # 增加调试计数器
        if self.training:
            self.debug_counter += 1

        is_lgc_only = (str(self.encoder).lower() == 'lightgcn') and (not self.use_ccf) and (not self.use_time_decay)

        # 1. Embedding selection with temporal fusion
        all_user_emb, all_item_emb = self.aggregate(edge_dropout=edge_dropout)

        user_gcn = all_user_emb[user.long()]
        pos_gcn = all_item_emb[positive.long()]
        neg_gcn = all_item_emb[negative.long()]

        # LightGCN-only path
        if is_lgc_only:
            pos_scores = torch.sum(user_gcn * pos_gcn, dim=1)
            neg_scores = torch.sum(user_gcn * neg_gcn, dim=1)
            bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-12))

            # Use temporal-aware embeddings when computing reg (temporal fusion applied here if enabled)
            u_emb = self.user_embedding(user.long(), pos_times if self.use_temporal_fusion else None)
            pos_i_emb = self.item_embedding(positive.long(), pos_times if self.use_temporal_fusion else None)
            neg_i_emb = self.item_embedding(negative.long(), pos_times if self.use_temporal_fusion else None)

            reg_loss = (1 / 2) * (u_emb.norm(2).pow(2) +
                                  pos_i_emb.norm(2).pow(2) +
                                  neg_i_emb.norm(2).pow(2)) / float(user.shape[0])
            reg_loss = reg_loss * self.reg_lambda

            na_loss = torch.tensor(0.0, device=self.device)

            # 添加调试输出
            if self.training and self.debug_counter % self.debug_interval == 0:
                print(f"=== LightGCN-only路径 (step={self.debug_counter}) ===")
                print(f"BPR Loss: {bpr_loss.item():.4f}")
                print(f"Reg Loss: {reg_loss.item():.4f}")

            return [bpr_loss, reg_loss, na_loss]

        # MF or LightGCN + contrastive learning
        # Use temporal-aware embeddings for raw side (temporal fusion applied only in representation)
        user_raw = self.user_embedding(user.long(), pos_times if self.use_temporal_fusion else None).to(self.device)
        pos_raw = self.item_embedding(positive.long(), pos_times if self.use_temporal_fusion else None).to(self.device)
        neg_raw = self.item_embedding(negative.long(), pos_times if self.use_temporal_fusion else None).to(self.device)

        # ========== 时间权重计算 ==========
        if self.use_time_decay and pos_times is not None and not self.use_temporal_fusion:
            time_weight = self.compute_relative_time_weights(pos_times)
            # 添加调试输出
            if self.training and self.debug_counter % self.debug_interval == 0:
                print(
                    f"时间权重计算完成: 形状={time_weight.shape}, 范围={time_weight.min().item():.3f}-{time_weight.max().item():.3f}")
        else:
            time_weight = torch.ones(user.size(0), device=self.device)
            if self.training and self.debug_counter % self.debug_interval == 0:
                if not self.use_time_decay:
                    print("时间衰减未启用，使用均匀权重")
                elif pos_times is None:
                    print("时间数据为None，使用均匀权重")
                elif self.use_temporal_fusion:
                    print("使用时间融合，不使用时间衰减权重")
        # ========== 时间权重计算结束 ==========

        # BPR loss: combine raw & gcn scores (scheme B), but temporal fusion only affects representations,
        # and time-weighting for BPR is disabled when using temporal fusion (keeps BPR consistent).
        raw_score = torch.sum(user_raw * pos_raw, dim=1)
        gcn_score = torch.sum(user_gcn * pos_gcn, dim=1)

        w = getattr(self, 'mf_gcn_weight', 0.5)

        # BPR uses fused score; if temporal fusion is used, raw embeddings already contain time info.
        pos_scores = (1 - w) * gcn_score + w * raw_score

        # BPR loss (neg uses GCN-neg embedding as before)
        neg_scores = torch.sum(user_gcn * neg_gcn, dim=1)
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-12))

        # Regularization (on raw/temporal fused embeddings)
        reg_loss = 0.5 * (user_raw.norm(2).pow(2) / user_raw.shape[0] +
                          pos_raw.norm(2).pow(2) / pos_raw.shape[0] +
                          neg_raw.norm(2).pow(2) / neg_raw.shape[0]) * self.reg_lambda

        # NA / contrastive learning loss: uses representation-level temporal fusion if enabled.
        if self.use_ccf:
            # fuse raw + gcn representations (both may include temporal info if use_temporal_fusion)
            u_norm = F.normalize(user_raw + user_gcn, dim=1)
            p_norm = F.normalize(pos_raw + pos_gcn, dim=1)

            # 正样本余弦相似度（其实就是 dot，因为已经 normalize）
            pos_score_na = torch.sum(u_norm * p_norm, dim=1)

            pos_exp = torch.exp(pos_score_na / self.tau)

            sim_up = torch.matmul(u_norm, p_norm.t())
            batch_size = user.size(0)
            mask = torch.eye(batch_size, device=sim_up.device).bool()
            sim_up = sim_up.masked_fill(mask, -1e9)

            denom_sum = torch.sum(torch.exp(sim_up / self.tau), dim=1)

            raw_na_loss = -torch.log(pos_exp / (denom_sum + 1e-12))

            # ======== New: Fuzzy 置信度 + 相似度权重 μ（加强版） ========
            if self.use_fuzzy:
                # 用 (user, positive) 去置信度字典里查
                if self.confidence_dict is not None:
                    u_np = user.detach().cpu().numpy()
                    p_np = positive.detach().cpu().numpy()
                    conf_list = []
                    for u_id, i_id in zip(u_np, p_np):
                        conf = self.confidence_dict.get((int(u_id), int(i_id)), 0.5)  # 默认0.5
                        conf_list.append(conf)
                    conf_tensor = torch.tensor(conf_list, device=self.device, dtype=torch.float32)
                else:
                    # 没有置信度矩阵时，退化为常数0.5，只依赖相似度
                    conf_tensor = torch.ones_like(pos_score_na, device=self.device) * 0.5

                # 1) 先做一次 sigmoid，把 (置信度 + 相似度) 压到 (0,1)
                z = self.alpha * conf_tensor + self.beta * pos_score_na
                mu_raw = torch.sigmoid(z)  # mu_raw 大概在 [0.3, 0.9] 之间

                # 2) 以 batch 均值为中心做“零均值化”
                mu_centered = mu_raw - mu_raw.mean()  # 正样本高一点、差样本低一点

                # 3) 稍微放大一下偏差，让权重差距更明显一点
                fuzzy_scale = 2.0  # ← 想更强可以调到 3.0，想温和一点可以调到 1.5
                mu = 1.0 + fuzzy_scale * mu_centered  # 平均值仍然约等于 1

                # 4) 做个安全 clamp，避免极端样本权重太夸张
                mu = torch.clamp(mu, 0.3, 3.0)

            else:
                # 未启用 fuzzy 时，μ = 1，不影响原逻辑
                mu = torch.ones_like(pos_score_na, device=self.device)
            # ======== Fuzzy 权重计算完毕 ========

            # Apply time weighting & fuzzy weighting:
            # - 如果启用 temporal fusion：只用 μ，不再额外乘 time_weight（时间信息已经进 embedding 里）
            # - 否则：time_weight * μ 一起作为样本权重
            if self.use_temporal_fusion:
                weighted_na = raw_na_loss * mu
            else:
                weighted_na = raw_na_loss * time_weight * mu

            na_loss = torch.mean(weighted_na) * self.ssl_lambda

            # 调试输出
            if self.training and self.debug_counter % self.debug_interval == 0:
                print(f"对比学习损失: 原始={raw_na_loss.mean().item():.4f}, 加权后={na_loss.item():.4f}")
                if self.use_time_decay and not self.use_temporal_fusion:
                    print(
                        f"对比学习时间衰减: time_weight范围={time_weight.min().item():.3f}-{time_weight.max().item():.3f}")
                if self.use_fuzzy:
                    print(
                        f"Fuzzy置信度加权: mu范围={mu.min().item():.3f}-{mu.max().item():.3f}")
        else:
            na_loss = torch.tensor(0.0, device=self.device)

        return [bpr_loss, reg_loss, na_loss]

    def cal_loss(self, batch_data):
        """Calculate loss with temporal fusion support"""
        # safe unpack
        if len(batch_data) >= 3:
            users = batch_data[0]
            positives = batch_data[1]
            negatives = batch_data[2]
            pos_times = batch_data[3] if len(batch_data) >= 4 else None
        else:
            raise ValueError("batch_data must contain at least users, positives, negatives")

        # ensure tensors on correct device
        users = users.to(self.device)
        positives = positives.to(self.device)
        negatives = negatives.to(self.device)
        if pos_times is not None:
            pos_times = pos_times.to(self.device)

        # apply edge dropout only during training; generate() / eval will use 0.0
        edge_dropout_rate = self.edge_dropout_rate if self.training else 0.0

        loss_list = self.forward(users, positives, negatives, pos_times, edge_dropout=edge_dropout_rate)
        total_loss = sum(loss_list)
        loss_dict = {
            'bpr_loss': float(loss_list[0].item()),
            'reg_loss': float(loss_list[1].item()),
            'na_loss': float(loss_list[2].item())
        }
        return total_loss, loss_dict

    @torch.no_grad()
    def generate(self):
        """Generate final embeddings for evaluation"""
        if self.encoder == 'MF' or self.adj_mat is None and self.edges is None:
            # Use base embeddings for evaluation (no temporal fusion applied here)
            user_emb = self.user_embedding.base_embedding.weight.detach().to(self.device)
            item_emb = self.item_embedding.base_embedding.weight.detach().to(self.device)
        else:
            user_emb, item_emb = self.aggregate(edge_dropout=0.0)
        return user_emb, item_emb

    def rating(self, user_emb, item_emb=None):
        """Compute rating scores for evaluation"""
        if item_emb is None:
            all_user, all_item = self.aggregate(edge_dropout=0.0)
            u = user_emb.long().to(self.device)
            u_gcn = all_user[u]
            score = torch.matmul(u_gcn, all_item.t())
            return score
        else:
            return torch.matmul(user_emb, item_emb.t())
