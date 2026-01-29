from utils.parse_args import args
import torch
import torch.optim as optim
from tqdm import tqdm
from utils.metrics import Metric
from os import path
import numpy as np
from utils.logger import log_exceptions
import time


class Trainer(object):
    def __init__(self, dataset, logger, pre_dataset=None):
        self.dataloader = dataset
        self.pre_dataloader = pre_dataset
        self.metric = Metric()
        self.logger = logger
        self.best_perform = {'recall': [0.], 'ndcg': [0.]}

    def create_optimizer(self, model):
        """
        只优化 requires_grad=True 的参数
        - 对我们新的 MF 残差模型很重要：base_emb 是 requires_grad=False，不会被更新
        - 对原来的 GraphPro 模型也兼容
        """
        params = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer = optim.Adam(params, lr=args.lr)

    def train_epoch(self, model, epoch_idx):
        self.dataloader.shuffle()
        s = 0
        loss_log_dict = {}
        ep_loss = 0
        time_1 = time.time()

        model.train()
        pbar = tqdm(total=self.dataloader.num_edges // args.batch_size + 1)
        while s + args.batch_size <= self.dataloader.num_edges:
            batch_data = self.dataloader.get_train_batch(s, s + args.batch_size)
            self.optimizer.zero_grad()
            loss, loss_dict = model.cal_loss(batch_data)
            loss.backward()
            self.optimizer.step()
            ep_loss += loss.item()

            # 记录各项 loss
            for loss_name in loss_dict:
                _loss_val = float(loss_dict[loss_name]) / (self.dataloader.num_edges // args.batch_size + 1)
                if loss_name not in loss_log_dict:
                    loss_log_dict[loss_name] = _loss_val
                else:
                    loss_log_dict[loss_name] += _loss_val

            s += args.batch_size
            pbar.update(1)
            if self.stop_flag:
                break

        time_2 = time.time()
        loss_log_dict['train_time'] = round(time_2 - time_1, 2)
        self.logger.log_loss(epoch_idx, loss_log_dict)

    @log_exceptions
    def train(self, model):
        """
        原始 pretrain 用的接口，不动
        """
        self.create_optimizer(model)
        self.best_perform = {'recall': [0.], 'ndcg': [0.]}
        self.stop_counter = 0
        self.stop_flag = False

        for epoch_idx in range(args.num_epochs):
            self.train_epoch(model, epoch_idx)
            self.evaluate(model, epoch_idx, self.dataloader)
            if self.stop_flag:
                break
        return self.best_perform

    @log_exceptions
    def train_finetune(self, model, pre_model=None):
        """
        微调阶段：
        - 兼容原 GraphPro：如果 pre_model 不为 None，就在 pre_dataloader / dataloader 上评估一下预训练模型；
        - 兼容新的 MF 残差微调：可以直接传 pre_model=None，此时跳过预训练模型的评估。
        """
        # 可选：验证预训练模型性能
        if pre_model is not None:
            pre_model.eval()
            # 如果你还想看预训练模型在 pretrain dataset 上的表现，可以取消下面注释：
            # if self.pre_dataloader is not None:
            #     self.logger.log("Testing Pretrained Model on Pretrain Dataset...")
            #     self.logger.log_eval(self.metric.eval(pre_model, self.pre_dataloader), self.metric.k)
            #
            # self.logger.log("Testing Pretrained Model on Test Dataset...")
            # self.logger.log_eval(self.metric.eval(pre_model, self.dataloader), self.metric.k)

        # 为微调模型创建优化器（只覆盖可训练参数）
        self.create_optimizer(model)
        self.best_perform = {'recall': [0.], 'ndcg': [0.]}
        self.stop_counter = 0
        self.stop_flag = False

        for epoch_idx in range(args.num_epochs):
            self.train_epoch(model, epoch_idx)
            self.evaluate(model, epoch_idx, self.dataloader)
            if self.stop_flag:
                break
        return self.best_perform

    def evaluate(self, model, epoch_idx, dataloader):
        model.eval()
        eval_result = self.metric.eval(model, dataloader)
        self.logger.log_eval(eval_result, self.metric.k)

        perform = eval_result['recall'][0]
        if perform > self.best_perform['recall'][0]:
            self.best_perform = eval_result
            self.logger.log(
                'Find better model at epoch: {}: recall={}'.format(
                    epoch_idx, self.best_perform['recall'][0]
                )
            )
            if args.log:
                self.save_model(model)
                self.logger.log('Model saved!')
            self.stop_counter = 0
        else:
            self.stop_counter += 1
            if self.stop_counter >= args.early_stop_patience:
                self.logger.log('Early stop!')
                self.logger.log(
                    f"Best performance: recall={self.best_perform['recall'][0]}, "
                    f"ndcg={self.best_perform['ndcg'][0]}"
                )
                self.stop_flag = True
        model.train()

    def save_model(self, model):
        self.save_path = path.join(args.save_dir, f'saved_model_{args.exp_time}.pt')
        torch.save(model.state_dict(), self.save_path)
        pass
