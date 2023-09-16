##########################################################
#
# Copyright (C) 2023-PRESENT: Keivan Ipchi Hagh
#
# Email:            keivanipchihagh@gmail.com
# GitHub:           https://github.com/keivanipchihagh
#
##########################################################

# Standard
import random
import torch
from torch import nn
from torch.utils.data import DataLoader

# Third-party
from .utils import plot_recon, plot_random_recon


class CallBack():

    def __init__(
            self,
            train_loader: DataLoader,
            valid_loader: DataLoader,
            train_n: int = 5,
            valid_n: int = 3,
            random_times: int = 5,
        ) -> 'CallBack':
        """
            CallBack

            Parameters:
                train_loader (DataLoader): Training data loader
                valid_loader (DataLoader): Validation data loader
                train_n (int): Number of training sample images
                valid_n (int): Number of validation sample images
                random_times (int): Number of validation random reconstructions
        """
        self.random_times = random_times
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # Random sample IDs
        train_idxs = random.sample(range(len(train_loader.dataset)), train_n)
        valid_idxs = random.sample(range(len(valid_loader.dataset)), valid_n)

        # Random sample images
        self.train_images = [self.train_loader.dataset[idx] for idx in train_idxs]
        self.valid_images = [self.valid_loader.dataset[idx] for idx in valid_idxs]


    def on_train_end(self, model: nn.Module, filename: str = None) -> None:
        """
            Callback to plot training data reconstruction

            Parameters:
                model (nn.Module): Model to use
                filename (str): Filename to save plot into
            Returns:
                None
        """
        plot_recon(
            model = model, 
            images = self.train_images,
            title = "Training Data Reconstruction",
            filename = filename
        )


    def on_valid_end(self, model: nn.Module, filename: str = None) -> None:
        """
            Callback to plot validation data random reconstructions

            Parameters:
                model (nn.Module): Model to use
                filename (str): Filename to save plot into
            Returns:
                None
        """
        plot_random_recon(
            model = model, 
            images = self.valid_images,
            title = "Validation Data Random Reconstructions",
            filename = filename,
            times = self.random_times,
        )


    def on_epoch_end(self, model: nn.Module, filename: str) -> None:
        """
            Callback to checkpoint model weights

            Parameters:
                model (nn.Module): Model to use
                filename (str): The name and path for which to save the weights
            Returns:
                None
        """
        torch.save(
            obj = model.state_dict(),
            f = filename
        )