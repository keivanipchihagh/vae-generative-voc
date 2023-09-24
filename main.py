##########################################################
#
# Copyright (C) 2023-PRESENT: Keivan Ipchi Hagh
#
# Email:            keivanipchihagh@gmail.com
# GitHub:           https://github.com/keivanipchihagh
#
##########################################################

# Standard
import os
import torch
from torch import nn
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

# Third-party
from train import Trainer
from model.model import VAE
from model.loss import KLMSE_Loss
from utils.logging import Logger
from data.loader import load_data
from utils.utils import setup_seed, get_model_params


def load_args() -> dict:
    """
        Loads CMD Arguments

        Returns:
            (dict): Arguments
    """
    parser = ArgumentParser(description='')
    # Global
    parser.add_argument('--use_cuda',       type=bool,  default=True,           help="Run on CUDA (default: True)")
    parser.add_argument('--seed',           type=int,   default=42,             help="Random Seed (default: 42)")
    parser.add_argument('--image_size',     type=int,   default=64,             help="Image Size (default: 64)")
    parser.add_argument('--tensorboard',    type=bool,  default=True,           help="Use Tensorboard (default: True)")
    # Model
    parser.add_argument('--latent_dim',     type=int,   default=512,            help="Latent Dimension (default: 128)")
    parser.add_argument('--kl_alpha',       type=int,   default=1,              help="Kullback Leibler coefficient (default: 1)")
    # Datasets
    parser.add_argument('--batch_size',     type=int,   default=16,             help="Batch Size (default: 16)")
    parser.add_argument('--shuffle',        type=bool,  default=True,           help="Shuffle Dataset (default: True)")
    parser.add_argument('--num_workers',    type=int,   default=2,              help="Number of Workers (defualt: 4)")
    # Training
    parser.add_argument('--max_epochs',     type=int,   default=1000,           help="Maximum Number of Epochs (default: 1000)")
    parser.add_argument('--resume',         type=str,   default=None,           help="Checkpoint to resume from (default: None)")
    parser.add_argument('--optim',          type=str,   default="adam",         help="Checkpoint to resume from (default: None)", choices=["sgd", "adam"])
    parser.add_argument('--lr',             type=float, default=5e-4,           help="Initial Learning Rate")
    parser.add_argument('--lr_schedule',    type=str,   default="poly",         help="Learning Rate Scheduler Policy (default: poly)", choices = ["poly"])
    parser.add_argument('--save_plots',     type=bool,  default=True,           help="Only save the plots to files")
    parser.add_argument('--n_images',       type=int,   default=1000,           help="Number of images to load for training and validation")

    return parser.parse_args()



if __name__ == '__main__':
    args = load_args()      # Load CMD Arguments

    # Create dirs if not already exist
    os.makedirs(f'results/images', exist_ok = True)
    os.makedirs(f'results/weights', exist_ok = True)
    os.makedirs(f'results/tensorboard', exist_ok = True)
    os.makedirs(f'results/psnrs', exist_ok = True)

    # Initalize Tensorboard
    tb_writer = None
    if args.tensorboard:
        tb_writer = SummaryWriter(f"results/tensorboard")

    # Enable Logging
    logger = Logger()

    # --- CUDA ---
    device = torch.device('cpu')
    if args.use_cuda:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info("CUDA:\tEnabled")
        else:
            logger.warn("CUDA:\tNot available!")
            args.use_cuda = False   # Disable CUDA forcefully

    # --- Set Seed ---
    setup_seed(args.seed)
    logger.info(f"Seed:\t{args.seed}")


    # --- Load Data ---
    train_loader, valid_loader = load_data(
        train_dirs = ["data\VOCdevkit\VOC2012\JPEGImages"],
        valid_dirs = ["data\VOCdevkit\VOC2007\JPEGImages"],
        batch_size = args.batch_size,
        image_size = args.image_size,
        num_workers = args.num_workers
    )
    logger.info(f"Dataset:\t{len(train_loader.dataset)}, {len(valid_loader.dataset)}")


    # --- Define Model ---
    model = VAE(
        in_channels = 3,
        latent_dim = 256,
        hidden_dims = [32, 64, 126, 256, 512],
        device = device
    )
    if args.use_cuda:
        model = nn.DataParallel(model).cuda()
    logger.info(f"Params:\t{get_model_params(model)}")


    # --- Define criterion ---
    criteria = KLMSE_Loss(
        kl_alpha = args.kl_alpha
    )
    if args.use_cuda:
        criteria = criteria.cuda()


    # --- Optimization Strategy ----
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(
            params = filter(lambda p: p.requires_grad, model.parameters()),
            lr = args.lr,
            momentum = 0.9,
            weight_decay = 1e-4
        )
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(
            params = filter(lambda p: p.requires_grad, model.parameters()),
            lr = args.lr,
            betas = (0.9, 0.999),
            eps = 1e-08,
            weight_decay = 1e-4
        )
    logger.info(f"Optimizer:\t{args.optim}")


    # --- Training ---
    trainer = Trainer(
        model = model,
        criteria = criteria,
        optimizer = optimizer,
        device = device
    )
    trainer.run(
        start_epoch = 0,
        end_epoch = args.max_epochs,
        train_loader = train_loader,
        valid_loader = valid_loader,
        tb_writer = tb_writer,
        save_plot = args.save_plots,
    )
