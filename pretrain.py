# pretrain.py
import sys
sys.path.append('./')

import random
import numpy as np
import torch
import importlib
from os import path
import setproctitle

from utils.parse_args import args
from utils.logger import Logger, log_exceptions
from utils.trainer import Trainer
# ensure your data class path matches below import; if your file is utils/dataloader.py change accordingly
from utils.dataloader import GraphProCCFData
setproctitle.setproctitle('GraphPro')

modules_class = 'modules.' if not args.plugin else 'modules.plugins.'

def import_model():
    module = importlib.import_module(modules_class + args.model)
    return getattr(module, args.model)

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

init_seed(args.seed)
logger = Logger(args)

if args.phase.startswith("pretrain"):
    @log_exceptions
    def run():
        edgelist_dataset = GraphProCCFData(args, phase='pretrain')
        model = import_model()(edgelist_dataset, args, phase='pretrain').to(args.device)

        trainer = Trainer(edgelist_dataset, logger)
        trainer.train(model)
    run()
