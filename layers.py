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


class Encoder(nn.Module):

    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            negative_slope: float = 0.01
        ) -> None:
        super(Encoder, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward Phase

            Parameters:
                x (torch.Tensor): Input Tensor
            Returns:
                (torch.Tensor): Output Tensor
        """
        return self.block(x)



class Decoder(nn.Module):

    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            negative_slope: float = 0.01
        ) -> None:
        super(Decoder, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward Phase

            Parameters:
                x (torch.Tensor): Input Tensor
            Returns:
                (torch.Tensor): Output Tensor
        """
        return self.block(x)



class OutLayer(nn.Module):

    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            negative_slope: float = 0.01
        ) -> None:
        super(OutLayer, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope),
            nn.Conv2d(
                in_channels=out_ch,
                out_channels= 3,
                kernel_size= 3,
                padding= 1
            ),
            nn.Tanh()
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward Phase

            Parameters:
                x (torch.Tensor): Input Tensor
            Returns:
                (torch.Tensor): Output Tensor
        """
        return self.block(x)