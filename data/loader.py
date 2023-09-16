##########################################################
#
# Copyright (C) 2023-PRESENT: Keivan Ipchi Hagh
#
# Email:            keivanipchihagh@gmail.com
# GitHub:           https://github.com/keivanipchihagh
#
##########################################################

# Standard
from pathlib import Path
from typing import List, Tuple
from torch.utils.data import DataLoader

# Third-party
from .dataset import VOCDataset


def load_data(
        train_dirs: List[str],
        valid_dirs: List[str],
        batch_size: int = 32,
        image_size: int = 64,
        num_workers: int = 4,
    ) -> Tuple[DataLoader, DataLoader]:
    """
        Loads the training and validation data

        Parameters:
            train_dirs (List[str]): List of filenames for training images
            test_dirs (List[str]): List of filenames for testing images
            batch_size (int): Number of samples per batch to load
            image_size (int): Height and width of each image
            num_workers (int): Number of workers to use
        Returns:
            (Tuple[DataLoader, DataLoader]): training and validation loaders
    """
    def _(dirs: List[str]):
        files = []
        [files.extend([str(file) for file in Path(dir).glob("*.jpg")]) for dir in dirs]
        dataset = VOCDataset(files, image_size)
        loader = DataLoader(
            dataset,                    # The dataset to load from
            batch_size = batch_size,    # How many samples per batch to load
            shuffle = True,             # Whether to shuffle the data
            drop_last = False,          # Drop the last incomplete batch
            num_workers = num_workers,  # Number of processes to use for loading the data
            pin_memory = True,          # avoid one implicit CPU-to-CPU copy
        )
        return loader

    train_loader = _(train_dirs)        # Load training data
    valid_loader = _(valid_dirs)        # Load validation data
    return train_loader, valid_loader
