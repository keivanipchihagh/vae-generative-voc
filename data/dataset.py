##########################################################
#
# Copyright (C) 2023-PRESENT: Keivan Ipchi Hagh
#
# Email:            keivanipchihagh@gmail.com
# GitHub:           https://github.com/keivanipchihagh
#
##########################################################

# Standard
import torch
import torchvision
from typing import List
from torch.utils.data import Dataset
from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
    ToPILImage
)


class VOCDataset(Dataset):

    def __init__(self, files: List[str], size: int) -> 'VOCDataset':
        """
            Reads a list of image paths and defines lazy transformations

            Parameters:
                files (List[str]): List of image pathes
                size (int): Image size (width and height)
        """
        self.files = files
        self.transformations = Compose([
            ToPILImage("RGB"),
            Resize((size, size)),
            ToTensor(),
        ])


    def __len__(self) -> int:
        """
            Returns number of images

            Returns:
                (int): Number of files on the dataset
        """
        return len(self.files)


    def __getitem__(self, i: int) -> torch.Tensor:
        """
            Reads and returns an image given the index

            Parameters:
                i (int): The index for which to return the image
            Returns:
                (Tensor): Image Tensor
        """
        img = torchvision.io.read_image(self.files[i])  # Read Image
        img = self.transformations(img)                 # Apply transformations
        if img.shape[0] == 1:
            img = torch.cat([img] * 3)
        return img
