# utils/graphproccf_data.py
import random
import numpy as np
import scipy.sparse as sp
import torch
from copy import deepcopy
from collections import defaultdict
import logging
from utils.parse_args import args
import pandas as pd

logger = logging.getLogger('GraphProCCFData')
logging.basicConfig(level=logging.INFO)


class GraphProCCFData:
    """
    高性能、兼容 EdgeListData 接口的 GraphPro-CCF 数据类

    功能:
      - 加载 pretrain.txt / pretrain_val.txt（与原 repo 格式一致）
      - 生成稀疏 user-item graph (scipy.sparse.coo_matrix)
      - 提供 edge_time_dict (双向, item index 在 dict 中为 item + num_users)
      - 支持 shuffle(), get_train_batch(start,end) 按边索引 (现在返回 time batch)
      - 支持 get_contrastive_batch(batch_size, num_negatives) (LightCCF 风格，返回对应时间)
      - 实现 LightCCF 批量负采样(lightccf_negative_sampling)
      - 兼容原仓库中字段名（train_user_dict / train_dict / test_user_dict / test_dict）
      - 额外：预计算 user_seq_items / user_seq_mask，用于微调阶段的 Attention+GRU
    """

    def __init__(self, args_obj, *, phase='pretrain', pre_dataset=None,
                 user_hist_files=None, has_time=True,
                 train_file=None, test_file=None):
        """
        args_obj: argparse.Namespace (来自 utils.parse_args.args)
        phase: 'pretrain' 或 'finetune'
        pre_dataset: 预训练 dataset（若用于 finetune），用于用户/物品对齐
        user_hist_files: finetune 时用于扩展 user history 的文件列表
        has_time: 是否读取第三列时间戳
        train_file: 训练文件路径（可选，默认为 None）
        test_file: 测试文件路径（可选，默认为 None）
        """
        if user_hist_files is None:
            user_hist_files = []

        self.args = args_obj
        self.phase = phase
        self.pre_dataset = pre_dataset
        self.has_time = has_time
        # 用于时间划分的小时间隔
        self.hour_interval = getattr(self.args, 'hour_interval_pre', 1) if phase == 'pretrain' else \
            getattr(self.args, 'hour_interval_f', 1)

        # containers
        self.train_user_dict = {}  # {user: [item,...]}
        self.test_user_dict = {}   # {user: [item,...]}
        # alias
        self.train_dict = self.train_user_dict
        self.test_dict = self.test_user_dict

        self.edgelist = []     # list of (user, item)
        self.edge_time = []    # parallel list of timestamps (floats)

        self.edge_time_dict = defaultdict(dict)  # user -> {item+num_users: time}
        self.user_hist_dict = {}

        # counts (filled later)
        self.num_users = 0
        self.num_items = 0
        self.num_edges = 0

        # fast structures
        self._edgelist_np = None
        self._edge_time_np = None

        # prompt/new edges holder (optional)
        self.new_edges = None

        # 如果未提供文件路径，使用默认路径
        if train_file is None:
            train_file = f"{self.args.data_path}/pretrain.txt"
        if test_file is None:
            test_file = f"{self.args.data_path}/pretrain_val.txt"

        # 1) 从文件 / DataFrame 加载
        self._load_from_files(train_file, test_file)

        # 2) 构建 user-item 稀疏图 (num_users, num_items)
        self.graph = sp.coo_matrix(
            (np.ones(self.num_edges, dtype=np.float32),
             (self._edgelist_np[:, 0], self._edgelist_np[:, 1])),
            shape=(self.num_users, self.num_items)
        )

        # 3) 构建 edge_time_dict（双向）
        if self.has_time:
            self.edge_time_dict = defaultdict(dict)
            for idx in range(self.num_edges):
                u = int(self._edgelist_np[idx, 0])
                item_id = int(self._edgelist_np[idx, 1])
                t = float(self._edge_time_np[idx])
                self.edge_time_dict[u][item_id + self.num_users] = t
                self.edge_time_dict[item_id + self.num_users][u] = t

        # 4) user_list & user_hist_dict
        self.user_list = list(self.train_user_dict.keys())
        if self.phase == 'pretrain':
            self.user_hist_dict = self.train_user_dict
        else:
            self.user_hist_dict = deepcopy(self.train_user_dict)
            if user_hist_files:
                self._load_user_hist_from_files(user_hist_files)

        # 5) 预计算用户最近 L 个交互序列（给微调 Attention+GRU 用）
        seq_len = getattr(self.args, "ft_seq_len", 10)
        self._build_user_seq_cache(seq_len)

        # 6) shuffle 初始化
        self.shuffle()

        logger.info(
            f"数据加载完成: {self.num_users} users, {self.num_items} items, "
            f"{self.num_edges} edges (phase={self.phase})"
        )

    # -------------------------
    # file loader (fast)
    # -------------------------
    def _load_from_files(self, train_file, test_file):
        """
        从文件或DataFrame加载数据
        支持两种数据源：
          - 文件路径 (字符串): 从指定txt文件读取
          - pandas.DataFrame: 直接从DataFrame对象加载
        """
        # 判断是否是 DataFrame
        is_train_df = hasattr(train_file, 'columns') and hasattr(train_file, 'iloc')
        is_test_df = hasattr(test_file, 'columns') and hasattr(test_file, 'iloc')

        # --- 训练数据 ---
        if is_train_df:
            train_df = train_file
            for _, row in train_df.iterrows():
                u = int(row['user'])
                items_str = row['item'] if pd.notna(row['item']) else ""
                times_str = row['time'] if 'time' in row and pd.notna(row.get('time')) else ""

                items = items_str.split() if items_str else []
                times = times_str.split() if times_str else ["0"] * len(items)

                if len(times) < len(items):
                    times.extend(["0"] * (len(items) - len(times)))

                items_int = [int(x) for x in items] if items else []
                times_float = [float(x) for x in times[:len(items)]] if items else []

                self.train_user_dict[u] = items_int
                for it, tt in zip(items_int, times_float):
                    self.edgelist.append((u, it))
                    self.edge_time.append(tt)
        else:
            with open(train_file, 'r', encoding='utf-8') as fr:
                for line in fr:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t')
                    if self.has_time and len(parts) >= 3:
                        u = int(parts[0])
                        items = parts[1].split()
                        times = parts[2].split()
                    else:
                        u = int(parts[0])
                        items = parts[1].split()
                        times = ["0"] * len(items)

                    items_int = [int(x) for x in items]
                    times_float = [float(x) for x in times]
                    self.train_user_dict[u] = items_int
                    for it, tt in zip(items_int, times_float):
                        self.edgelist.append((u, it))
                        self.edge_time.append(tt)

        # --- 测试数据 ---
        if is_test_df:
            test_df = test_file
            for _, row in test_df.iterrows():
                u = int(row['user'])
                items_str = row['item'] if pd.notna(row['item']) else ""
                items = items_str.split() if items_str else []
                items_int = [int(x) for x in items] if items else []
                self.test_user_dict[u] = items_int
        else:
            with open(test_file, 'r', encoding='utf-8') as fr:
                for line in fr:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t')
                    u = int(parts[0])
                    items = [int(x) for x in parts[1].split()]
                    self.test_user_dict[u] = items

        # --- 统计用户/物品数 ---
        if self.pre_dataset is not None:
            self.num_users = self.pre_dataset.num_users
            self.num_items = self.pre_dataset.num_items
        else:
            max_user_train = max(self.train_user_dict.keys()) if self.train_user_dict else -1
            max_user_test = max(self.test_user_dict.keys()) if self.test_user_dict else -1
            self.num_users = max(max_user_train, max_user_test) + 1

            max_item_train = max([max(lst) for lst in self.train_user_dict.values()]) if self.train_user_dict else -1
            max_item_test = max([max(lst) for lst in self.test_user_dict.values()]) if self.test_user_dict else -1
            self.num_items = max(max_item_train, max_item_test) + 1

        # --- 转 numpy ---
        if len(self.edgelist) > 0:
            self._edgelist_np = np.array(self.edgelist, dtype=np.int32)
        else:
            self._edgelist_np = np.zeros((0, 2), dtype=np.int32)

        if len(self.edge_time) > 0:
            times_np = np.array(self.edge_time, dtype=np.float64)
            times_step = self._timestamp_to_time_step(times_np)
            self._edge_time_np = times_step.astype(np.float32)
        else:
            self._edge_time_np = np.zeros((len(self.edgelist),), dtype=np.float32)
        self.num_edges = self._edgelist_np.shape[0]

    def sparse_adjacency_matrix(self):
        """生成 LightCCF 原版稀疏邻接矩阵"""
        row = []
        col = []
        data = []

        for u, i in zip(self._edgelist_np[:, 0], self._edgelist_np[:, 1]):
            row.append(u)
            col.append(self.num_users + i)
            data.append(1.0)
            # 双向边
            row.append(self.num_users + i)
            col.append(u)
            data.append(1.0)

        mat = sp.coo_matrix(
            (data, (row, col)),
            shape=(self.num_users + self.num_items, self.num_users + self.num_items),
            dtype=np.float32
        )
        return mat

    # -------------------------
    # time quantization helper
    # -------------------------
    def _timestamp_to_time_step(self, timestamps):
        if timestamps.size == 0:
            return timestamps
        least = timestamps.min()
        interval_sec = max(1.0, float(self.hour_interval)) * 3600.0
        steps = (timestamps - least) / interval_sec
        return steps

    # -------------------------
    # user-hist files loader (finetune)
    # -------------------------
    def _load_user_hist_from_files(self, files):
        for file in files:
            with open(file, 'r', encoding='utf-8') as fr:
                for line in fr:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t')
                    u = int(parts[0])
                    items = [int(x) for x in parts[1].split()]
                    if u in self.user_hist_dict:
                        self.user_hist_dict[u].extend(items)
                    else:
                        self.user_hist_dict[u] = items

    # -------------------------
    # 预计算用户最近 L 个交互序列 (for finetune Attention+GRU)
    # -------------------------
    def _build_user_seq_cache(self, seq_len: int):
        """
        为每个 user 预先构造:
          - self.user_seq_items: [num_users, seq_len] LongTensor
          - self.user_seq_mask:  [num_users, seq_len] BoolTensor (True 表示该位置有真实 item)
        规则：
          - 使用 user_hist_dict[u] 的“最近 seq_len 条记录”
          - 右对齐：前面 pad 0/False，后面是真实 item/True
        """
        self.seq_len = seq_len
        num_users = self.num_users

        user_seq_items = torch.zeros((num_users, seq_len), dtype=torch.long)
        user_seq_mask = torch.zeros((num_users, seq_len), dtype=torch.bool)

        for u, hist in self.user_hist_dict.items():
            if len(hist) == 0:
                continue
            tail = hist[-seq_len:]
            tail_len = len(tail)
            pad = seq_len - tail_len
            if pad < 0:
                tail = tail[-seq_len:]
                tail_len = len(tail)
                pad = 0
            user_seq_items[u, pad:pad + tail_len] = torch.tensor(tail, dtype=torch.long)
            user_seq_mask[u, pad:pad + tail_len] = True

        self.user_seq_items = user_seq_items
        self.user_seq_mask = user_seq_mask

        logger.info(
            f"已预计算用户序列缓存: user_seq_items.shape={self.user_seq_items.shape}, "
            f"user_seq_mask.shape={self.user_seq_mask.shape}, seq_len={seq_len}"
        )

    # -------------------------
    # shuffle / batching
    # -------------------------
    def shuffle(self):
        """
        Shuffle edgelist order and user_list for per-edge/per-user batching.
        Uses numpy permutation for speed.
        """
        if self.num_edges > 0:
            perm = np.random.permutation(self.num_edges)
            self._edgelist_np = self._edgelist_np[perm]
            self._edge_time_np = self._edge_time_np[perm]
            self.edgelist = [(int(x), int(y)) for x, y in self._edgelist_np.tolist()]
            self.edge_time = [float(x) for x in self._edge_time_np.tolist()]

        self.user_list = list(self.train_user_dict.keys())
        random.shuffle(self.user_list)
        self.index_map = {i: u for i, u in enumerate(self.user_list)}

    # -------------------------
    # LightCCF negative sampling (batch style)
    # -------------------------
    def lightccf_negative_sampling(self, batch_users, batch_pos_items, num_negatives=1):
        """
        LightCCF 负采样：
         - batch_pos_items: iterable of positive items
         - batch_users: iterable of users
        返回: torch.LongTensor(len(batch_users) * num_negatives)
        """
        if isinstance(batch_pos_items, torch.Tensor):
            batch_pos_list = batch_pos_items.cpu().tolist()
        elif isinstance(batch_pos_items, np.ndarray):
            batch_pos_list = batch_pos_items.tolist()
        else:
            batch_pos_list = list(batch_pos_items)

        pool = list(set(batch_pos_list))
        pool_len = len(pool)
        batch_neg = []

        for u in batch_users:
            u_int = int(u)
            user_pos_set = set(self.train_user_dict.get(u_int, []))
            for _ in range(num_negatives):
                candidate = None
                if pool_len > 0:
                    for _ in range(100):
                        neg = pool[random.randint(0, pool_len - 1)]
                        if neg not in user_pos_set:
                            candidate = neg
                            break
                if candidate is None:
                    while True:
                        neg = random.randint(0, self.num_items - 1)
                        if neg not in user_pos_set:
                            candidate = neg
                            break
                batch_neg.append(candidate)

        return torch.LongTensor(batch_neg).to(args.device)

    # -------------------------
    # get_train_batch: 按边索引（EdgeList 风格）
    # -------------------------
    def get_train_batch(self, start, end):
        """
        按 edge index 切片返回 (users, pos_items, neg_items, pos_times)
        """
        if start >= self.num_edges:
            return (torch.LongTensor([]).to(args.device),
                    torch.LongTensor([]).to(args.device),
                    torch.LongTensor([]).to(args.device),
                    torch.FloatTensor([]).to(args.device))
        end = min(end, self.num_edges)
        slice_np = self._edgelist_np[start:end]
        users = torch.LongTensor(slice_np[:, 0]).to(args.device)
        pos_items = torch.LongTensor(slice_np[:, 1]).to(args.device)

        neg_items = []
        for u, _ in slice_np:
            u_int = int(u)
            user_pos = set(self.train_user_dict.get(u_int, []))
            while True:
                neg = random.randint(0, self.num_items - 1)
                if neg not in user_pos:
                    neg_items.append(neg)
                    break
        neg_items = torch.LongTensor(neg_items).to(args.device)

        pos_times_np = self._edge_time_np[start:end]
        pos_times = torch.from_numpy(np.array(pos_times_np, dtype=np.float32)).to(args.device)

        return users, pos_items, neg_items, pos_times

    # -------------------------
    # get_contrastive_batch: 用户批次 (LightCCF 风格)
    # -------------------------
    def get_contrastive_batch(self, batch_size):
        """
        返回 (users, pos_items_list, neg_items, pos_times_list)
        """
        all_users = list(self.train_user_dict.keys())
        if len(all_users) == 0:
            return (torch.LongTensor([]).to(args.device),
                    [], torch.LongTensor([]).to(args.device), [])

        batch_users = random.sample(all_users, min(batch_size, len(all_users)))
        pos_items_list = [self.train_user_dict[u] for u in batch_users]

        first_pos_items = [items[0] for items in pos_items_list if len(items) > 0]
        neg_items = self.lightccf_negative_sampling(batch_users, first_pos_items)

        pos_times_list = []
        all_times = []
        for u, pos_list in zip(batch_users, pos_items_list):
            times = []
            for it in pos_list:
                global_item = it + self.num_users
                t = self.edge_time_dict[u].get(global_item, 0.0)
                times.append(t)
                all_times.append(t)
            pos_times_list.append(times)

        if len(all_times) > 0:
            min_t, max_t = min(all_times), max(all_times)
            range_t = max(max_t - min_t, 1e-6)
            for i in range(len(pos_times_list)):
                pos_times_list[i] = [(t - min_t) / range_t for t in pos_times_list[i]]
                if len(set(pos_times_list[i])) == 1:
                    pos_times_list[i] = [t + 1e-6 * idx for idx, t in enumerate(pos_times_list[i])]

        users_t = torch.LongTensor(batch_users).to(args.device)
        neg_items = torch.LongTensor(neg_items).to(args.device)
        return users_t, pos_items_list, neg_items, pos_times_list

    # -------------------------
    # utility
    # -------------------------
    @property
    def edgelist_np(self):
        return self._edgelist_np

    @property
    def edge_time_np(self):
        return self._edge_time_np

    def build_prompt_graph(self):
        """
        构建 prompt 图：Gp_sample ⊕ Gnew
        """
        device = torch.device(args.device if args.device else 'cpu')

        if isinstance(self.graph, torch.Tensor):
            edges_p = self.graph
        elif hasattr(self.graph, '_indices'):
            edges_p = self.graph._indices().t()
        else:
            edges_p = torch.LongTensor(self.graph)

        total_edges = edges_p.size(0)
        num_sample = int(total_edges * getattr(args, "prompt_ratio", 0.2))
        if num_sample <= 0:
            num_sample = min(1, total_edges)
        perm = torch.randperm(total_edges)[:num_sample]
        edges_p_sample = edges_p[perm]

        edges_new = getattr(self, 'new_edges', None)
        if edges_new is None:
            edges_new = torch.empty(0, 2, dtype=torch.long)

        if edges_p_sample.numel() == 0:
            edges_prompt = edges_new
        elif edges_new.numel() == 0:
            edges_prompt = edges_p_sample
        else:
            edges_prompt = torch.cat([edges_p_sample, edges_new], dim=0)

        edge_norm = torch.ones(edges_prompt.size(0), device=device)
        return edges_prompt.to(device), edge_norm


# ==== 放在 utils/dataloader.py 里，GraphProCCFData 定义之后 ====
class GraphProCCFData_FT(GraphProCCFData):
    """
    微调阶段专用 DataLoader：
    - 不改动原来的 GraphProCCFData，避免影响预训练 / 原有训练逻辑
    - 只重写 get_train_batch，逻辑仍然基于：
        * self.edgelist:  每一行是 (user, item)
        * self.edge_time: 与 edgelist 一一对应的时间戳
        * self.train_user_dict: user -> set(正样本 item)，用于负采样
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_train_batch(self, start, end):
        def negative_sampling(user_item, train_user_set, n=1):
            neg_items = []
            for user, _ in user_item:
                user = int(user)
                for i in range(n):
                    while True:
                        neg_item = np.random.randint(low=0, high=self.num_items, size=1)[0]
                        if neg_item not in train_user_set[user]:
                            break
                    neg_items.append(neg_item)
            return neg_items

        ui_pairs = self.edgelist[start:end]

        # ★ 如果你想保留空 batch 检查，用 len 替代 shape
        if len(ui_pairs) == 0:
            # 按你的需求处理：可以 raise / return None / 直接跳过
            # 暂时可以直接 raise，看看是不是有逻辑问题
            raise ValueError(f"Empty batch: start={start}, end={end}, num_edges={self.num_edges}")

        # 如果 ui_pairs 可能是 list，这里统一转成 numpy
        if isinstance(ui_pairs, list):
            ui_pairs = np.array(ui_pairs, dtype=np.int32)

        users = torch.LongTensor(ui_pairs[:, 0]).to(args.device)
        pos_items = torch.LongTensor(ui_pairs[:, 1]).to(args.device)

        if args.model == "MixGCF":
            neg_items = negative_sampling(ui_pairs, self.train_user_dict, args.n_negs)
        else:
            neg_items = negative_sampling(ui_pairs, self.train_user_dict, 1)

        neg_items = torch.LongTensor(neg_items).to(args.device)

        return users, pos_items, neg_items


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger('train_logger')
logger.setLevel(logging.INFO)


class EdgeListData:
    def __init__(self, train_file, test_file, phase='pretrain', pre_dataset=None, user_hist_files=[], has_time=True):
        logger.info(f"Loading dataset for {phase}...")
        self.phase = phase
        self.has_time = has_time
        self.pre_dataset = pre_dataset

        self.hour_interval = args.hour_interval_pre if phase == 'pretrain' else args.hour_interval_f

        self.edgelist = []
        self.edge_time = []
        self.num_users = 0
        self.num_items = 0
        self.num_edges = 0

        self.train_user_dict = {}
        self.test_user_dict = {}

        self._load_data(train_file, test_file, has_time)

        if phase == 'pretrain':
            self.user_hist_dict = self.train_user_dict
        elif phase == 'finetune':
            self.user_hist_dict = deepcopy(self.train_user_dict)
            self._load_user_hist_from_files(user_hist_files)

        users_has_hist = set(list(self.user_hist_dict.keys()))
        all_users = set(list(range(self.num_users)))
        users_no_hist = all_users - users_has_hist
        logger.info(f"Number of users from all users with no history: {len(users_no_hist)}")
        for u in users_no_hist:
            self.user_hist_dict[u] = []

    def _read_file(self, train_file, test_file, has_time=True):
        with open(train_file, 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                if not has_time:
                    user, items = line[:2]
                    times = " ".join(["0"] * len(items.split(" ")))
                else:
                    user, items, times = line

                for i in items.split(" "):
                    self.edgelist.append((int(user), int(i)))
                for i in times.split(" "):
                    self.edge_time.append(int(i))
                self.train_user_dict[int(user)] = [int(i) for i in items.split(" ")]

        self.test_edge_num = 0
        with open(test_file, 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                user, items = line[:2]
                self.test_user_dict[int(user)] = [int(i) for i in items.split(" ")]
                self.test_edge_num += len(self.test_user_dict[int(user)])
        logger.info('Number of test users: {}'.format(len(self.test_user_dict)))

    def _read_pd(self, train_pd, test_pd, has_time=True):
        for i in range(len(train_pd)):
            line = train_pd.iloc[i]
            if not has_time:
                user, items = line[0], line[1]
                times = " ".join(["0"] * len(items.split(" ")))
            else:
                user, items, times = line[0], line[1], line[2]

            for i in items.split(" "):
                self.edgelist.append((int(user), int(i)))
            for i in times.split(" "):
                self.edge_time.append(int(i))
            self.train_user_dict[int(user)] = [int(i) for i in items.split(" ")]

        self.test_edge_num = 0
        for i in range(len(test_pd)):
            line = test_pd.iloc[i]
            user, items = line[0], line[1]
            self.test_user_dict[int(user)] = [int(i) for i in items.split(" ")]
            self.test_edge_num += len(self.test_user_dict[int(user)])
        logger.info('Number of test users: {}'.format(len(self.test_user_dict)))

    def _load_data(self, train_file, test_file, has_time=True):
        if isinstance(train_file, pd.DataFrame):
            self._read_pd(train_file, test_file, has_time)
        else:
            self._read_file(train_file, test_file, has_time)

        self.edgelist = np.array(self.edgelist, dtype=np.int32)
        # refine timestamp to predefined time steps at intervals
        # 0 as padding for self-loop
        self.edge_time = 1 + self.timestamp_to_time_step(np.array(self.edge_time, dtype=np.int32))
        self.num_edges = len(self.edgelist)
        if self.pre_dataset is not None:
            self.num_users = self.pre_dataset.num_users
            self.num_items = self.pre_dataset.num_items
        else:
            self.num_users = max([np.max(self.edgelist[:, 0]) + 1, np.max(list(self.test_user_dict.keys())) + 1])
            self.num_items = max([np.max(self.edgelist[:, 1]) + 1,
                                  np.max([np.max(self.test_user_dict[u]) for u in self.test_user_dict.keys()]) + 1])

        logger.info('Number of users: {}'.format(self.num_users))
        logger.info('Number of items: {}'.format(self.num_items))
        logger.info('Number of edges: {}'.format(self.num_edges))

        self.graph = sp.coo_matrix((np.ones(self.num_edges), (self.edgelist[:, 0], self.edgelist[:, 1])),
                                   shape=(self.num_users, self.num_items))

        if self.has_time:
            self.edge_time_dict = defaultdict(dict)
            for i in range(len(self.edgelist)):
                self.edge_time_dict[self.edgelist[i][0]][self.edgelist[i][1] + self.num_users] = self.edge_time[i]
                self.edge_time_dict[self.edgelist[i][1] + self.num_users][self.edgelist[i][0]] = self.edge_time[i]
                # self.edge_time_dict[self.edgelist[i][0]][self.edgelist[i][0]] = 0
                # self.edge_time_dict[self.edgelist[i][1]][self.edgelist[i][1]] = 0

        # homogenous edges generation
        # if args.ab in ["homo", "full"]:
        #     self.ii_adj = self.graph.T.dot(self.graph).tocoo()
        #     # sort the values in the sparse matrix and get top x% values and corresponding edges
        #     percentage_ii = 0.01
        #     tmp_data = sorted(self.ii_adj.data, reverse=True)
        #     tmp_data_xpercent, tmp_data_xpercent_len = tmp_data[int(len(tmp_data) * percentage_ii)], int(len(tmp_data) * percentage_ii)
        #     self.ii_adj.data = np.where(self.ii_adj.data > tmp_data_xpercent, self.ii_adj.data, 0)
        #     self.ii_adj.eliminate_zeros()
        #     logger.info(f"Sampled {len(self.ii_adj.data)} i-i edges from all {len(tmp_data)} edges.")

        #     self.uu_adj = self.graph.dot(self.graph.T).tocoo()
        #     # same filtering for uu_adj
        #     percentage_uu = 0.01
        #     tmp_data = sorted(self.uu_adj.data, reverse=True)
        #     tmp_data_xpercent, tmp_data_xpercent_len = tmp_data[int(len(tmp_data) * percentage_uu)], int(len(tmp_data) * percentage_uu)
        #     self.uu_adj.data = np.where(self.uu_adj.data > tmp_data_xpercent, self.uu_adj.data, 0)
        #     self.uu_adj.eliminate_zeros()
        #     logger.info(f"Sampled {len(self.uu_adj.data)} u-u edges from all {len(tmp_data)} edges.")

        # self.graph = nx.Graph()
        # for i in range(len(self.edgelist)):
        #     self.graph.add_edge(self.edgelist[i][0], self.edgelist[i][1], time=self.edge_time[i])
        # print(self.graph.number_of_nodes(), self.graph.number_of_edges())
        # self.ui_adj = nx.adjacency_matrix(self.graph, weight=None)[:self.num_users, self.num_users:].tocoo()
        # self.ii_adj = self.ui_adj.T.dot(self.ui_adj).tocoo()
        # self.uu_adj = self.ui_adj.dot(self.ui_adj.T).tocoo()

        # self.graph = Data(torch.zeros(self.num_users + self.num_items, 1), torch.tensor(self.edgelist).t(), torch.tensor(self.edge_time))

    def _load_user_hist_from_files(self, user_hist_files):
        for file in user_hist_files:
            with open(file, 'r') as f:
                for line in f:
                    line = line.strip().split('\t')
                    user, items = int(line[0]), [int(i) for i in line[1].split(" ")]
                    try:
                        self.user_hist_dict[user].extend(items)
                    except KeyError:
                        self.user_hist_dict[user] = items

    def sample_subgraph(self):
        pass

    def get_train_batch(self, start, end):

        def negative_sampling(user_item, train_user_set, n=1):
            neg_items = []
            for user, _ in user_item:
                user = int(user)
                for i in range(n):
                    while True:
                        neg_item = np.random.randint(low=0, high=self.num_items, size=1)[0]
                        if neg_item not in train_user_set[user]:
                            break
                    neg_items.append(neg_item)
            return neg_items

        ui_pairs = self.edgelist[start:end]
        users = torch.LongTensor(ui_pairs[:, 0]).to(args.device)
        pos_items = torch.LongTensor(ui_pairs[:, 1]).to(args.device)
        if args.model == "MixGCF":
            neg_items = negative_sampling(ui_pairs, self.train_user_dict, args.n_negs)
        else:
            neg_items = negative_sampling(ui_pairs, self.train_user_dict, 1)
        neg_items = torch.LongTensor(neg_items).to(args.device)
        # ⭐ 新增：把边的时间也取出来
        # self.edge_time 在 _load_data 里已经做过 timestamp_to_time_step 和 +1 处理
        pos_times_np = self.edge_time[start:end]  # 这是 numpy array 或 list
        pos_times = torch.from_numpy(np.array(pos_times_np, dtype=np.float32)).to(args.device)
        return users, pos_items, neg_items, pos_times

    def shuffle(self):
        random_idx = np.random.permutation(self.num_edges)
        self.edgelist = self.edgelist[random_idx]
        self.edge_time = self.edge_time[random_idx]

    def _generate_binorm_adj(self, edgelist):
        adj = sp.coo_matrix((np.ones(len(edgelist)), (edgelist[:, 0], edgelist[:, 1])),
                            shape=(self.num_users, self.num_items), dtype=np.float32)

        a = sp.csr_matrix((self.num_users, self.num_users))
        b = sp.csr_matrix((self.num_items, self.num_items))
        adj = sp.vstack([sp.hstack([a, adj]), sp.hstack([adj.transpose(), b])])
        adj = (adj != 0) * 1.0
        degree = np.array(adj.sum(axis=-1))
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        adj = adj.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()

        ui_adj = adj.tocsr()[:self.num_users, self.num_users:].tocoo()
        return adj

    def timestamp_to_time_step(self, timestamp_arr, least_time=None):
        interval_hour = self.hour_interval
        if least_time is None:
            least_time = np.min(timestamp_arr)
            print("Least time: ", least_time)
            print("2nd least time: ", np.sort(timestamp_arr)[1])
            print("3rd least time: ", np.sort(timestamp_arr)[2])
            print("Max time: ", np.max(timestamp_arr))
        timestamp_arr = timestamp_arr - least_time
        timestamp_arr = timestamp_arr // (interval_hour * 3600)
        return timestamp_arr


if __name__ == '__main__':
    edgelist_dataset = EdgeListData("dataset/yelp_small/train.txt", "dataset/yelp_small/test.txt")