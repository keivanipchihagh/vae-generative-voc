##########################################################
#
# Copyright (C) 2023-PRESENT: Keivan Ipchi Hagh
#
# Email:            keivanipchihagh@gmail.com
# GitHub:           https://github.com/keivanipchihagh
#
##########################################################

# Standard
from argparse import ArgumentParser

# Third-party
from model.model import VAE
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

    return parser.parse_args()



if __name__ == '__main__':
    args = load_args()      # Load CMD Arguments

    # Enable Logging
    logger = Logger()

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
        use_cuda = True
    )
    logger.info(f"Params:\t{get_model_params(model)}")
