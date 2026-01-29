import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='LightCCF Pretraining and GraphPro-compatible parser')

    # ------------------- 通用参数 ------------------- #
    parser.add_argument('--phase', type=str, default='pretrain', help='pretrain / finetune')
    parser.add_argument('--plugin', action='store_true', default=False, help='use plugin mode')
    parser.add_argument('--save_path', type=str, default='results', help='where to save models and logs')
    parser.add_argument('--data_path', type=str, default='dataset/douban-book', help='dataset folder')
    parser.add_argument('--exp_name', type=str, default='exp1', help='experiment name')
    parser.add_argument('--desc', type=str, default='', help='experiment description')
    parser.add_argument('--log', type=int, default=1, help='whether to log')
    parser.add_argument('--device', type=str, default='cuda', help='device to use')
    parser.add_argument('--seed', type=int, default=2023)

    # ------------------- LightCCF 参数 ------------------- #
    parser.add_argument('--embedding_size', type=int, default=64, help='user/item embedding size')
    parser.add_argument('--gcn_layer', type=int, default=3, help='number of GCN layers')
    parser.add_argument('--encoder', type=str, default='LightGCN', help='encoder type (LightGCN/MF)')
    parser.add_argument('--ssl_lambda', type=float, default=5.0, help='SSL loss weight')
    parser.add_argument('--tau', type=float, default=0.28, help='temperature for NA loss')
    parser.add_argument('--reg_lambda', type=float, default=0.0001, help='regularization weight')

    # ------------------- 训练参数 ------------------- #
    parser.add_argument('--train_epoch', type=int, default=300)
    parser.add_argument('--train_batch_size', type=int, default=2048)
    parser.add_argument('--learn_rate', type=float, default=0.001)
    parser.add_argument('--test_frequency', type=int, default=1)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--sparsity_test', type=int, default=0)

    # ------------------- GraphPro / 兼容参数 ------------------- #
    parser.add_argument('--model', type=str, default='LightCCF', help='model to use')
    parser.add_argument('--pre_model', type=str, default='LightCCF', help='pretrained model')
    parser.add_argument('--f_model', type=str, default='LightCCF', help='fine-tune model')
    parser.add_argument('--pre_model_path', type=str, default='pretrained_model.pt')
    parser.add_argument('--num_layers', type=int, default=3, help='number of GNN layers, compatible with GraphPro')
    parser.add_argument('--batch_size', type=int, default=2048, help='GraphPro统一 batch size')
    parser.add_argument('--eval_batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001, help='GraphPro learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--early_stop_patience', type=int, default=10)
    parser.add_argument('--neighbor_sample_num', type=int, default=5)
    parser.add_argument('--neg_num', type=int, default=1)

    return parser

# ------------------- 可直接使用 ------------------- #
args = parse_args().parse_args()
# 兼容 model 自动赋值逻辑
if args.pre_model == args.f_model:
    args.model = args.pre_model
elif args.pre_model != 'LightGCN':
    args.model = args.pre_model
