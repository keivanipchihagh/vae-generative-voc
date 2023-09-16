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
from typing import Tuple
import torch.nn.functional as F


# Wikipedia:    https://en.wikipedia.org/wiki/Kullback-Leibler_divergence
class KullbackLeibler(nn.Module):

    def __init__(
            self,
        ) -> "KullbackLeibler":
        """ Kullback Leibler (KL) Divergence """
        super().__init__()


    def forward(self, logvar: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        """
            Forward Phase

            Parameters:
                logvar (torch.Tensor): Log of Variance
                mean (torch.Tensor): Mean
            Returns:
                (torch.Tensor): Loss
        """
        kl = (logvar ** 2 + mean ** 2 - torch.log(logvar) - 1/2).sum()  # Calculate KL
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



class KLMSE_Loss(nn.Module):

    def __init__(self, kl_alpha: float = 0.1) -> 'KLMSE_Loss':
        """
            KL-MSE Loss

            Parameters:
                kl_alpha (float): Kullback Leibler coefficient
        """
        super().__init__()

        self.kl_alpha = kl_alpha
        self.kl = KullbackLeibler()
        self.mse = MSE()


    def forward(
        self,
        target: torch.Tensor,
        output: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
            Forward Phase
            
            Parameters:
                target (torch.Tensor): Desired image
                output (torch.Tensor): Output image of the model
                mean (torch.Tensor): mean
                logvar (torch.Tensor): Log of Variance
            Returns:
                (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): KL, MSE and total loss
        """
        _kl = self.kl(logvar, mean) * self.kl_alpha # KL
        _mse = self.mse(output, target)             # MSE
        return _kl, _mse, (_kl + _mse)
