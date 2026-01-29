import torch
import numpy as np
from utils.parse_args import args
import time
import logging  # 添加logger导入

# 初始化logger
logger = logging.getLogger('train_logger')


class Metric(object):
    def __init__(self):
        self.metrics = args.metrics.split(";")
        self.k = [int(k) for k in args.metrics_k.split(";")]

    def recall(self, test_data, r, k):
        right_pred = r[:, :k].sum(1)
        recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
        recall = np.sum(right_pred / recall_n)
        return recall

    def precision(self, r, k):
        right_pred = r[:, :k].sum(1)
        precis_n = k
        precision = np.sum(right_pred) / precis_n
        return precision

    def mrr(self, r, k):
        pred_data = r[:, :k]
        scores = np.log2(1. / np.arange(1, k + 1))
        pred_data = pred_data / scores
        pred_data = pred_data.sum(1)
        return np.sum(pred_data)

    def ndcg(self, test_data, r, k):
        assert len(r) == len(test_data)
        pred_data = r[:, :k]

        test_matrix = np.zeros((len(pred_data), k))
        for i, items in enumerate(test_data):
            length = k if k <= len(items) else len(items)
            test_matrix[i, :length] = 1
        max_r = test_matrix
        idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
        dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
        dcg = np.sum(dcg, axis=1)
        idcg[idcg == 0.] = 1.
        ndcg = dcg / idcg
        ndcg[np.isnan(ndcg)] = 0.
        return np.sum(ndcg)

    def get_label(self, test_data, pred_data):
        r = []
        for i in range(len(test_data)):
            ground_true = test_data[i]
            predict_topk = pred_data[i]
            pred = list(map(lambda x: x in ground_true, predict_topk))
            pred = np.array(pred).astype("float")
            r.append(pred)
        return np.array(r).astype('float')

    def eval_batch(self, data, topks):
        sorted_items = data[0].numpy()
        ground_true = data[1]
        r = self.get_label(ground_true, sorted_items)

        result = {}
        for metric in self.metrics:
            result[metric] = []

        for k in topks:
            for metric in result:
                if metric == 'recall':
                    result[metric].append(self.recall(ground_true, r, k))
                if metric == 'ndcg':
                    result[metric].append(self.ndcg(ground_true, r, k))
                if metric == 'precision':
                    result[metric].append(self.precision(r, k))
                if metric == 'mrr':
                    result[metric].append(self.mrr(r, k))

        for metric in result:
            result[metric] = np.array(result[metric])

        return result

    def eval(self, model, dataloader):
        time_1 = time.time()
        result = {}
        for metric in self.metrics:
            result[metric] = np.zeros(len(self.k))

        test_user_set = dataloader.test_user_dict
        user_hist_dict = dataloader.user_hist_dict

        test_users = list(test_user_set.keys())
        n_test_users = len(test_users)
        n_user_batchs = n_test_users // args.eval_batch_size + 1

        with torch.no_grad():
            user_emb, item_emb = model.generate()

        batch_ratings = []
        ground_truths = []
        test_user_count = 0
        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * args.eval_batch_size
            end = (u_batch_id + 1) * args.eval_batch_size

            user_list_batch = test_users[start: end]
            user_batch = torch.LongTensor(np.array(user_list_batch)).to(args.device)
            u_g_embeddings = user_emb[user_batch]
            # predict result
            # all-item test
            i_g_embddings = item_emb
            with torch.no_grad():
                batch_pred = model.rating(u_g_embeddings, i_g_embddings).cpu()
            # filter out history items - 修正参数顺序
            batch_pred = self._mask_history_pos(
                batch_pred, user_hist_dict, user_list_batch)  # 修正参数顺序
            _, batch_rate = torch.topk(batch_pred, k=max(self.k))
            batch_ratings.append(batch_rate.cpu())
            # ground truth at the first
            ground_truth_batch = []
            for uid in user_list_batch:
                ground_truth_batch.append(test_user_set[uid])
            ground_truths.append(ground_truth_batch)
            test_user_count += batch_pred.shape[0]
        assert test_user_count == len(test_user_set)

        # calculate metrics
        data_pair = zip(batch_ratings, ground_truths)
        eval_results = []
        for _data in data_pair:
            eval_results.append(self.eval_batch(_data, self.k))
        for batch_result in eval_results:
            for metric in self.metrics:
                result[metric] += batch_result[metric] / len(test_user_set)

        time_2 = time.time()
        result['eval_time'] = round(time_2 - time_1, 2)
        for metric in result:
            result[metric] = np.round(result[metric], 4)

        return result

    def eval_grouped(self, model, dataloader, group='tuned'):
        time_1 = time.time()
        result = {}
        for metric in self.metrics:
            result[metric] = np.zeros(len(self.k))

        tune_user_set = dataloader.train_user_dict
        test_user_set = dataloader.test_user_dict
        tuned_users = list(set(tune_user_set.keys()).intersection(set(test_user_set.keys())))
        untuned_users = list(set(test_user_set.keys()).difference(set(tune_user_set.keys())))
        user_hist_dict = dataloader.user_hist_dict

        if group == 'tuned':
            test_users = tuned_users
        else:
            test_users = untuned_users

        n_test_users = len(test_users)
        n_user_batchs = n_test_users // args.eval_batch_size + 1

        with torch.no_grad():
            user_emb, item_emb = model.generate()

        batch_ratings = []
        ground_truths = []
        test_user_count = 0
        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * args.eval_batch_size
            end = (u_batch_id + 1) * args.eval_batch_size

            user_list_batch = test_users[start: end]
            user_batch = torch.LongTensor(np.array(user_list_batch)).to(args.device)
            u_g_embeddings = user_emb[user_batch]
            # predict result
            # all-item test
            i_g_embddings = item_emb
            with torch.no_grad():
                batch_pred = model.rating(u_g_embeddings, i_g_embddings).cpu()
            # filter out history items - 修正参数顺序
            batch_pred = self._mask_history_pos(
                batch_pred, user_hist_dict, user_list_batch)  # 修正参数顺序
            _, batch_rate = torch.topk(batch_pred, k=max(self.k))
            batch_ratings.append(batch_rate.cpu())
            # ground truth at the first
            ground_truth_batch = []
            for uid in user_list_batch:
                ground_truth_batch.append(test_user_set[uid])
            ground_truths.append(ground_truth_batch)
            test_user_count += batch_pred.shape[0]
        assert test_user_count == len(test_users)

        # calculate metrics
        data_pair = zip(batch_ratings, ground_truths)
        eval_results = []
        for _data in data_pair:
            eval_results.append(self.eval_batch(_data, self.k))
        for batch_result in eval_results:
            for metric in self.metrics:
                result[metric] += batch_result[metric] / len(test_users)

        time_2 = time.time()
        result['eval_time'] = round(time_2 - time_1, 2)
        for metric in result:
            result[metric] = np.round(result[metric], 4)

        return result

    def _mask_history_pos(self, batch_pred, user_hist_dict, user_ids):
        """
        安全的掩码处理 - 修正版本
        batch_pred: 预测矩阵 [batch_size, num_items]
        user_hist_dict: 用户历史字典 {user_id: [item1, item2, ...]}
        user_ids: 用户ID列表 [user_id1, user_id2, ...]
        """
        # 确保user_hist_dict是字典类型
        if not isinstance(user_hist_dict, dict):
            logger.warning(f"user_hist_dict类型为{type(user_hist_dict)}，期望dict，尝试转换")
            try:
                # 尝试将列表或其他类型转换为字典
                if isinstance(user_hist_dict, list):
                    user_hist_dict = {i: hist for i, hist in enumerate(user_hist_dict)}
                else:
                    user_hist_dict = {}
            except Exception as e:
                logger.error(f"转换user_hist_dict失败: {e}, 使用空字典")
                user_hist_dict = {}

        # 批量处理每个用户的历史掩码
        for i, user_id in enumerate(user_ids):
            try:
                # 安全获取用户历史
                if isinstance(user_hist_dict, dict):
                    pos_list = user_hist_dict.get(user_id, [])
                else:
                    # 回退方案
                    pos_list = []
                    logger.debug(f"user_hist_dict不是字典类型，使用空历史")
            except Exception as e:
                logger.debug(f"获取用户{user_id}历史时出错: {e}, 使用空历史")
                pos_list = []

            # 如果用户有历史记录，进行掩码
            if pos_list:
                try:
                    # 确保pos_list是列表或可迭代对象
                    if not isinstance(pos_list, (list, tuple, np.ndarray)):
                        pos_list = [pos_list]

                    # 转换为numpy数组并确保是整数类型
                    pos_indices = np.array(pos_list, dtype=np.int32)

                    # 确保索引不越界
                    num_items = batch_pred.shape[1]
                    valid_indices = pos_indices[(pos_indices >= 0) & (pos_indices < num_items)]

                    if len(valid_indices) > 0:
                        batch_pred[i, valid_indices] = -np.inf

                except Exception as e:
                    logger.debug(f"掩码用户{user_id}历史时出错: {e}")

        return batch_pred