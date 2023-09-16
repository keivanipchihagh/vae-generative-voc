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
from torch import nn
import torch.nn.functional as F


# Wikipedia:    https://en.wikipedia.org/wiki/Kullback-Leibler_divergence
class KullbackLeibler(nn.Module):

    def __init__(
            self,
            alpha: float = 0.1,
        ) -> "KullbackLeibler":
        """
            Kullback Leibler (KL) Divergence

            Parameters:
                alpha (float): Alpha
        """
        super().__init__()
        self.alpha = alpha


    def forward(self, logvar: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        """
            Forward Phase

            Parameters:
                logvar (torch.Tensor): Logarithm of Variance
                mean (torch.Tensor): Mean
            Returns:
                (torch.Tensor): Loss
        """
        kl = (logvar ** 2 + mean ** 2 - torch.log(logvar) - 1/2).sum()  # Calculate KL
        kl *= self.alpha                                                # Apply coefficient
        return kl


# Wikipedia:    https://en.wikipedia.org/wiki/Mean_squared_error
class MSE(nn.Module):

    def __init__(self) -> None:
        super().__init__()


    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
            Forward Phase

            Parameters:
                output (torch.Tensor): Output image of the model
                target (torch.Tensor): Desired image
        """
        return F.mse_loss(output, target, reduction = 'sum')