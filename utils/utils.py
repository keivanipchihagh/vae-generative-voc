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
from typing import Union
from typing import List
from matplotlib import pyplot as plt


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



def calculate_psnr(image1: Union[np.array, torch.Tensor], image2: Union[np.array, torch.Tensor]):
    """
        Peak signal-to-noise ratio
        
        Parameters:
            image1 (Union[np.array, torch.Tensor]): First iamge
            image2 (Union[np.array, torch.Tensor]): Second iamge
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



def plot_recon(
        model: nn.Module,
        images: List[torch.Tensor],
        title: str,
        filename: str = None,
    ) -> None:
    """
        Plots reconstruction

        Parameters:
            model (VAE): The model to use
            images (DataLoader): List of image Tensors
            title (str): Title of the plot
            filename (str): If spesified, the plot will onyl be saved at the given path, displays it otherwise.
        Returns:
            None
    """
    model.eval()                # Set model in evaluation
    if filename: plt.ioff()     # Disable interactive mode
    plt.figure(figsize=(10, 3))
    plt.title(title)
    n = len(images)

    for i, image in enumerate(images):
        image = image.unsqueeze(0)

        with torch.no_grad():
            output, _, _ = model(image)

        # Plot original image
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image.cpu().squeeze().permute(1, 2, 0).numpy())
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Plot reconstructed image
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(output.cpu().squeeze().permute(1, 2, 0).numpy())
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Calculate PSNR
        psnr = calculate_psnr(image, output)
        ax.set_title(f'{psnr:0.2f}')

    if filename: plt.savefig(filename, bbox_inches='tight')
    else: plt.show()



def plot_random_recon(
        model: nn.Module,
        images: List[torch.Tensor],
        title: str,
        filename: str = None,
        times: int = 5,
    ) -> None:
    """
        Plots reconstruction

        Parameters:
            model (VAE): The model to use
            images (DataLoader): List of image Tensors
            title (str): Title of the plot
            filename (str): If spesified, the plot will onyl be saved at the given path, displays it otherwise.
            times (int): Number of times to perform
        Returns:
            None
    """
    model.eval()                # Set model in evaluation
    if filename: plt.ioff()     # Disable interactive mode
    plt.figure(figsize=(10, 3))
    plt.title(title)
    n = len(images)

    for i, image in enumerate(images):
        image = image.unsqueeze(0)
        
        # Plot original image
        ax = plt.subplot(n, times + 1, (i * (times + 1) + 1))
        plt.imshow(image.cpu().squeeze().permute(1, 2, 0).numpy())
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        ax.set_title('Original')

        for j in range(times):
            with torch.no_grad():
                output, _, _ = model(image)
            output = output.cpu().squeeze().permute(1, 2, 0).numpy()

            # Plot random reconstructed image
            ax = plt.subplot(n, times+1, (i * (times + 1)) + j + 2)
            plt.imshow(output)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Calculate PSNR
            psnr = calculate_psnr(output, image.cpu().squeeze().permute(1, 2, 0).numpy())
            ax.set_title(f'{psnr:0.2f}')

    if filename: plt.savefig(filename, bbox_inches='tight')
    else: plt.show()
