import os
import random
import sys
import time
import warnings

import numpy as np
import torch
from torch.backends import cudnn
from tqdm import tqdm, trange

from data import create_dataloader
from utils import util
from utils.logger import Logger
import wandb

def set_seed(seed):
    cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Trainer:
    def __init__(self, task):
        self.task = task
        from options.distill_options import DistillOptions as Options
        from distillers import create_distiller as create_model

        opt = Options().parse()

        opt.tensorboard_dir = opt.log_dir if opt.tensorboard_dir is None else opt.tensorboard_dir
        print(' '.join(sys.argv))
        if opt.phase != 'train':
            warnings.warn('You are not using training set for %s!!!' % task)
        with open(os.path.join(opt.log_dir, 'opt.txt'), 'a') as f:
            f.write(' '.join(sys.argv) + '\n')
        set_seed(opt.seed)                      # 设置随机种子

        dataloader = create_dataloader(opt)  # create a dataset given opt.dataset_mode and other options
        dataset_size = len(dataloader.dataset)  # get the number of images in the dataset.
        print('The number of training images = %d' % dataset_size)

        model = create_model(opt)  # create a model given opt.model and other options
        model.setup(opt)  # regular setup: load and print networks; create schedulers
        logger = Logger(opt)

        self.opt = opt
        self.dataloader = dataloader
        self.model = model
        self.logger = logger

    def evaluate(self, epoch, iter, message):
        start_time = time.time()
        metrics = self.model.evaluate_model(iter)
        self.logger.print_current_metrics(epoch, iter, metrics, time.time() - start_time)
        self.logger.plot(metrics, iter)
        self.logger.print_info(message)
        self.model.save_networks('latest')

    def start(self):
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        opt = self.opt
        model = self.model

        if self.opt.project:
            wandb.init(project=self.opt.project, name=self.opt.name)
            config = wandb.config
            for k, v in sorted(vars(opt).items()):
                setattr(config, k, v)

        util.load_network(model.netG_teacher_A, os.path.join(self.opt.checkpoints_path, 'best_A_net_G_A.pth'))
        util.load_network(model.netD_teacher_A, os.path.join(self.opt.checkpoints_path, 'best_A_net_D_A.pth'))
        util.load_network(model.netG_student, os.path.join(self.opt.checkpoints_path, '348_net_G.pth'))
        model.load_best_teacher()
        model.optimize_student_parameters()