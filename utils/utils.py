##########################################################
#
# Copyright (C) 2023-PRESENT: Keivan Ipchi Hagh
#
# Email:            keivanipchihagh@gmail.com
# GitHub:           https://github.com/keivanipchihagh
#
##########################################################

# Standard
import math
import torch
import random
import numpy as np
from torch import nn


def setup_seed(seed: int) -> None:
    """
        Sets the seed for generating numbers

        Parameters:
            seed (int): Seed Number
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)



def calculate_psnr(image1: np.array|torch.Tensor, image2: np.array|torch.Tensor):
    """
        Peak signal-to-noise ratio
        
        Parameters:
            image1 (np.array|torch.Tensor): First iamge
            image2 (np.array|torch.Tensor): Second iamge
        Returns:
            (float) the PSNR score
    """
    # Tensor to Numpy
    if isinstance(image1, torch.Tensor): image1 = image1.cpu()
    if isinstance(image2, torch.Tensor): image2 = image2.cpu()
    image1 = np.array(image1)
    image2 = np.array(image2)

    # Calculate the mean squared error (MSE)
    mse = np.mean((image1 - image2) ** 2)

    # Calculate the maximum possible pixel value
    max_pixel_value = np.max(image1)

    # Calculate the PSNR using the MSE and maximum pixel value
    psnr = 20 * math.log10(max_pixel_value / math.sqrt(mse))

    return psnr



def get_model_params(model: nn.Module) -> int:
    """
        Calculate total number of Model Parameters

        Parameters:
            model (nn.Module): Model
        Returns:
            (int): Number of Parameters
    """
    params = 0
    for parameter in model.parameters():
        param = 1
        for j in range(len(parameter.size())):
            param *= parameter.size(j)
        params += param
    return params
