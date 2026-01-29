import argparse
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='GraphPro')

    # ---------------- General ----------------
    parser.add_argument('--phase', type=str, default='pretrain')
    parser.add_argument('--plugin', action='store_true', default=False)
    parser.add_argument('--save_path', type=str, default="saved", help='where to save model and logs')
    parser.add_argument('--data_path', type=str, default="dataset/yelp", help='where to load data')
    parser.add_argument('--exp_name', type=str, default='1')
    parser.add_argument('--desc', type=str, default='')
    parser.add_argument('--ab', type=str, default='full')
    parser.add_argument('--log', type=int, default=1)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument(
        '--eps',
        type=float,
        default=0.1,
        help='perturbation strength epsilon for SimGCL'
    )
    parser.add_argument('--temp', type=float, default=0.2,
                        help='temperature for SGL contrastive loss')
    parser.add_argument('--lbd', type=float, default=1.0,
                        help='contrastive loss weight for SimGCL')
    # ---------------- Model ----------------
    parser.add_argument('--model', type=str, default='GraphPro', help='Final model to run')
    parser.add_argument('--pre_model', type=str, default='GraphPro', help='Pretrained model for finetune')
    parser.add_argument('--f_model', type=str, default='GraphPro', help='Finetune model')
    parser.add_argument('--pre_model_path', type=str, default='pretrained_model.pt')

    # ---------------- Training ----------------
    parser.add_argument('--hour_interval_pre', type=float, default=1)
    parser.add_argument('--hour_interval_f', type=int, default=1)
    parser.add_argument('--emb_dropout', type=float, default=0)
    parser.add_argument('--updt_inter', type=int, default=1)
    parser.add_argument('--samp_decay', type=float, default=0.05)
    parser.add_argument('--edge_dropout', type=float, default=0.5)
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--eval_batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--neighbor_sample_num', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--metrics', type=str, default='recall;ndcg')
    parser.add_argument('--metrics_k', type=str, default='20')
    parser.add_argument('--early_stop_patience', type=int, default=10)
    parser.add_argument('--neg_num', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)

    # ---------------- CL / LightCCF ----------------
    parser.add_argument('--tau', type=float, default=0.22)
    parser.add_argument('--ssl_lambda', type=float, default=5.0)
    parser.add_argument('--reg_lambda', type=float, default=0.0001)
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--gcn_layer', type=int, default=3)
    parser.add_argument('--encoder', type=str, default='MF')
    parser.add_argument('--use_ccf', action='store_true', default=False)
    parser.add_argument('--use_bpr_with_cl', action='store_true', default=False)
    parser.add_argument('--ccf_lambda', type=float, default=0.5)
    parser.add_argument('--lambda_t', type=float, default=0.0)
    parser.add_argument('--use_fuzzy', action='store_true', default=True,
                        help='启用模糊对比学习机制（结合置信度矩阵和余弦相似度）')
    parser.add_argument('--alpha', type=float, default=0,
                        help='模糊隶属度中置信度的权重（默认：0.3）')
    parser.add_argument('--beta', type=float, default=0,
                        help='模糊隶属度中余弦相似度的权重（默认：0.7）')
    # +++ 新增：时间衰减加权参数 +++
    parser.add_argument('--use_time_decay', action='store_true', default=False,
                        help='Enable time-aware weighting for contrastive learning')
    parser.add_argument('--time_decay_lambda', type=float, default=0.1,
                        help='Decay coefficient lambda for time weighting')
    parser.add_argument('--max_time_step', type=int, default=365,
                        help='Max time step for normalization')
    parser.add_argument('--time_base', type=str, choices=['current', 'min'], default='current',
                        help='Time baseline for decay calculation')

    # ---------------- 新增：表示层时间融合参数 ----------------
    parser.add_argument('--use_temporal_fusion', action='store_true', default=False,
                        help='Enable temporal fusion in embedding layer (recommended)')
    parser.add_argument('--temporal_fusion_alpha', type=float, default=0.1,
                        help='Fusion coefficient for temporal-aware embeddings (default: 0.1)')
    parser.add_argument('--temporal_fusion_type', type=str, choices=['linear', 'gated', 'multi_scale'],
                        default='linear',
                        help='Type of temporal fusion mechanism')
    parser.add_argument('--temporal_embed_dim', type=int, default=8,
                        help='Temporal embedding dimension for encoding time features')
    # 新增自适应温度参数
    parser.add_argument('--use_adaptive_tau', action='store_true', default=False,
                        help='Enable adaptive temperature scheduling based on time distribution')
    parser.add_argument('--tau_alpha', type=float, default=0.5,
                        help='Strength of temperature adaptation (default: 0.5)')
    parser.add_argument('--min_tau', type=float, default=0.05,
                        help='Minimum temperature value (default: 0.05)')
    parser.add_argument('--max_tau', type=float, default=0.5,
                        help='Maximum temperature value (default: 0.5)')
    parser.add_argument('--finetune_use_cl', action='store_true', help='Whether to use CL loss during finetune')
    parser.add_argument('--ft_ssl_lambda', type=float, default=0.1, help='CL loss weight during finetune')
    parser.add_argument("--ft_dropout", type=float, default=0.1)
    parser.add_argument("--ft_hidden_dim", type=int, default=256)
    parser.add_argument("--ft_lr", type=float, default=1e-4)
    parser.add_argument("--ft_num_layers", type=int, default=2)
    # ===== interpolation & PISA-mix (for finetune_mf_residual.py) =====
    parser.add_argument("--init_l2norm", type=int, default=1, help="L2 normalize init embeddings (0/1)")
    parser.add_argument("--interp_scheme", type=str, default="graphpro", choices=["graphpro", "recent"],
                        help="graphpro: newest weight smallest; recent: newest weight largest")

    parser.add_argument("--use_pisa_mix", type=int, default=0,
                        help="use PISA to perturb base interpolation weights (0/1)")
    # 兼容旧命令（不想改命令也能跑）
    parser.add_argument("--use_pisa_beta_perturb", type=int, default=None,
                        help="DEPRECATED alias of --use_pisa_mix (0/1)")

    parser.add_argument("--pisa_dir", type=str, default="None", help="dir of pisa_pref_weights")
    parser.add_argument("--pisa_strict", type=int, default=0, help="if 1, missing pisa files will raise error")

    # 强度控制（你要“强一点点”主要调 pisa_eta_mix / pisa_eps）
    parser.add_argument("--pisa_eta_mix", type=float, default=0.10,
                        help="max deviation for pretrain ratio (e.g. 0.10 => 0.5±0.1)")
    parser.add_argument("--pisa_eps", type=float, default=0.10,
                        help="temperature for similarity comparison (smaller => stronger bias)")

    parser.add_argument("--pisa_gamma", type=float, default=0.30, help="history internal perturb strength")
    parser.add_argument("--pisa_delta", type=float, default=0.30,
                        help="history internal perturb clamp (multiplier range)")

    parser.add_argument("--pisa_log_users", type=str, default="0,1,2,10,100",
                        help="comma-separated user ids to print weights")
    parser.add_argument(
        "--ft_gate_mode",
        type=str,
        default="learned",
        choices=["learned", "fixed"],
        help="Gate mode in GraphProMFResFinetune. "
             "learned: gate=sigmoid(trainable raw); "
             "fixed: gate=constant ft_gate_value."
    )
    parser.add_argument(
        "--ft_gate_value",
        type=float,
        default=1.0,
        help="Only used when ft_gate_mode=fixed. "
             "1.0 = disable gating but keep residual (base + delta); "
             "0.0 = disable residual (base only)."
    )
    # ===== residual gate switch (alias) =====
    parser.add_argument(
        "--gate_mode",
        type=str,
        default=None,
        choices=["learned", "one", "zero"],
        help="Alias for residual gate: learned=learnable sigmoid gate; "
             "one=fixed gate=1; zero=fixed gate=0. "
             "If you explicitly pass --ft_gate_mode, it has priority."
    )

    # ---------------- Finetune Alignment Weight (NEW) ----------------
    parser.add_argument(
        "--ft_align_lambda",
        type=float,
        default=0.0,
        help="Alignment regularization weight in finetune (PISA-style)."
    )

    return parser

# Optional args for specific models
def parse_args_simgcl(parser):
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--lbd', type=float, default=0.5)
    return parser

def parse_args_sgl(parser):
    parser.add_argument('--temp', type=float, default=0.2)
    parser.add_argument('--lbd', type=float, default=0.1)
    return parser

def parse_args_mixgcf(parser):
    parser.add_argument('--n_negs', type=int, default=16)
    return parser

def parse_args_dau(parser):
    parser.add_argument('--gamma', type=float, default=1)
    return parser

def parse_args_duo_emb(parser):
    parser.add_argument('--last_model_path', type=str, default=None)
    return parser
def parse_args_recdcl(parser):
    # --- RecDCL loss weights ---
    add_arg_if_absent(parser, "--all_bt_coeff", type=float, default=1.0,
                      help="overall weight of Barlow-Twins loss (feature-wise)")
    add_arg_if_absent(parser, "--bt_coeff", type=float, default=0.1,
                      help="off-diagonal weight inside BT loss")

    add_arg_if_absent(parser, "--poly_coeff", type=float, default=0.0,
                      help="weight of polynomial feature loss")
    add_arg_if_absent(parser, "--a", type=float, default=1.0)
    add_arg_if_absent(parser, "--polyc", type=float, default=0.0)
    add_arg_if_absent(parser, "--degree", type=int, default=2)

    add_arg_if_absent(parser, "--mom_coeff", type=float, default=1.0,
                      help="weight of momentum (BYOL-style) loss")
    add_arg_if_absent(parser, "--momentum", type=float, default=0.9,
                      help="momentum for target history update")

    # Optional: add BPR on top of RecDCL (default off)
    add_arg_if_absent(parser, "--recdcl_rec_coeff", type=float, default=0.0,
                      help="extra BPR weight for RecDCL (0=off)")

    return parser

# ---------------- Parse arguments ----------------
parser = parse_args()
args, unknown = parser.parse_known_args()
# utils/parse_args.py 里：在 parse_args() 内或外加这个 helper（保证不重复添加参数）
def add_arg_if_absent(parser, *name_or_flags, **kwargs):
    for f in name_or_flags:
        if f in parser._option_string_actions:
            return
    parser.add_argument(*name_or_flags, **kwargs)


# 在 parse_args() 函数里，所有已有 parser.add_argument(...) 后面、return 前面，加下面这一块：
add_arg_if_absent(parser, "--pretrain_weight", type=float, default=0.5)
add_arg_if_absent(parser, "--init_l2norm", type=int, default=1)

add_arg_if_absent(parser, "--interp_scheme", type=str, default="graphpro",
                  choices=["graphpro", "recent"])  # recent=只用上一阶段

# === PISA mix：影响原插值比例（不是替代） ===
add_arg_if_absent(parser, "--use_pisa_mix", type=int, default=0)     # 1 开启
add_arg_if_absent(parser, "--pisa_dir", type=str, default=None)
add_arg_if_absent(parser, "--pisa_strict", type=int, default=0)      # 1: 缺文件直接报错

add_arg_if_absent(parser, "--pisa_eta_mix", type=float, default=0.35)  # 强度（越大越明显）
add_arg_if_absent(parser, "--pisa_eps", type=float, default=0.02)
add_arg_if_absent(parser, "--pisa_use_delta", type=int, default=1)     # per-user 去均值

# === 自动放大：保证能看到 0.45/0.55 这种变化 ===
add_arg_if_absent(parser, "--pisa_auto_scale", type=int, default=1)
add_arg_if_absent(parser, "--pisa_target_ratio", type=float, default=0.55)  # 想更强就 0.60
add_arg_if_absent(parser, "--pisa_scale_q", type=float, default=0.5)        # 更强就 0.3
add_arg_if_absent(parser, "--pisa_scale_max", type=float, default=2000.0)
add_arg_if_absent(parser, "--pisa_sim_clip", type=float, default=4.0)

# === invalid 用户也跟着动一点点（小幅） ===
add_arg_if_absent(parser, "--pisa_invalid_fill", type=str, default="stage_mean",
                  choices=["none", "stage_mean"])
add_arg_if_absent(parser, "--pisa_min_dev_invalid", type=float, default=0.02)  # invalid 最小偏移
add_arg_if_absent(parser, "--pisa_invalid_alpha", type=float, default=0.2)

add_arg_if_absent(parser, "--cgcl_neg_mode", type=str, default="inbatch",
                  choices=["full", "inbatch", "sampled"])
add_arg_if_absent(parser, "--cgcl_neg_k_u", type=int, default=4096)
add_arg_if_absent(parser, "--cgcl_neg_k_i", type=int, default=4096)

add_arg_if_absent(parser, "--cgcl_ssl_temp", type=float, default=0.2)
add_arg_if_absent(parser, "--cgcl_alpha", type=float, default=0.5)
add_arg_if_absent(parser, "--cgcl_beta", type=float, default=0.5)
add_arg_if_absent(parser, "--cgcl_gamma", type=float, default=0.5)
add_arg_if_absent(parser, "--cgcl_reg_alpha", type=float, default=1.0)
add_arg_if_absent(parser, "--cgcl_reg_beta", type=float, default=1.0)
add_arg_if_absent(parser, "--cgcl_reg_gamma", type=float, default=1.0)

# ====================================================
# ---------------- Fix model name: command line --model has priority ----------------
# If command line specifies --model explicitly,保持它；否则按旧逻辑处理
if '--model' not in unknown:
    if args.pre_model == args.f_model:
        args.model = args.pre_model
    elif args.pre_model != 'LightGCN':
        args.model = args.pre_model

# Add optional model-specific args
if args.model == 'SimGCL':
    parser = parse_args_simgcl(parser)
elif args.model == 'SGL':
    parser = parse_args_sgl(parser)
elif args.model == 'MixGCF':
    parser = parse_args_mixgcf(parser)
elif args.model == 'DirectAU':
    parser = parse_args_dau(parser)
elif args.model == 'RecDCL':
    parser = parse_args_recdcl(parser)
if args.phase.startswith("duo_emb_"):
    parser = parse_args_duo_emb(parser)

# Final parse to update args
args = parser.parse_args()
if args.pisa_dir is None:
    args.pisa_dir = os.path.join(args.data_path, "pisa_pref_weights")
# ===== map --gate_mode to --ft_gate_mode/--ft_gate_value (compat) =====
if getattr(args, "gate_mode", None) is not None and ("--ft_gate_mode" not in sys.argv):
    if args.gate_mode == "learned":
        args.ft_gate_mode = "learned"
    elif args.gate_mode == "one":
        args.ft_gate_mode = "fixed"
        args.ft_gate_value = 1.0
    elif args.gate_mode == "zero":
        args.ft_gate_mode = "fixed"
        args.ft_gate_value = 0.0
