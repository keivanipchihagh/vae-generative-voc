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
from typing import Tuple, List

# Third-party
from .layers import Encoder, Decoder, OutLayer


class VAE(nn.Module):

    def __init__(
            self,
            in_channels: int,
            latent_dim: int,
            hidden_dims: List[int] = [32, 64, 128, 256, 512],
            device: torch.device = torch.device('cpu')
        ) -> 'VAE':
        super(VAE, self).__init__()
        self.device = device

        self.latent_dim = latent_dim

        modules = []

        # Encoder
        for h_dim in hidden_dims:
            modules.append(Encoder(in_channels, h_dim))
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)

        # Latent Space
        self.mean_layer = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(Decoder(hidden_dims[i], hidden_dims[i + 1]))

        self.decoder = nn.Sequential(*modules)
        self.final_layer = OutLayer(hidden_dims[-1], hidden_dims[-1])


    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
            Reparameterization trick to sample from N(mu, var) from N(0,1)

            Parameters:
                mean (torch.Tensor): Mean of the latent Gaussian
                logvar (torch.Tensor): Standard deviation of the latent Gaussian
            Returns:
                (torch.Tensor): Output tensor
        """
        std = torch.exp(0.5 * logvar)                       # Standard Deviation
        eps = torch.randn_like(std, device=self.device)     # Distribution
        return mean + (eps * std)                           # Sample


    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Forward pass for the encoder

            Parameters:
                x (torch.torch.Tensor): Input
            Returns:
                (Tuple[torch.Tensor, torch.Tensor]): mean and logvar Tensors
        """
        x = self.encoder(x)                 # Encode
        x = torch.flatten(x, start_dim=1)   # Flatten
        mu = self.mean_layer(x)             # Mean
        logvar = self.logvar_layer(x)       # Logvar
        return mu, logvar


    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward pass for the decoder

            Parameters:
                x (torch.torch.Tensor): Input
            Returns:
                (torch.torch.Tensor): Reconstructed image
        """
        x = self.decoder_input(x)   # Unflatten
        x = x.view(-1, 512, 2, 2)   # Reshape
        x = self.decoder(x)         # Decode
        x = self.final_layer(x)     # Final Layer
        return x


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
            Forward pass for the VAE

            Parameters:
                x (torch.torch.Tensor): Input image
            Returns:
                (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Reconstructed image, mean and logvar
        """
        mu, logvar = self.encode(x)           # Encode
        z = self.reparameterize(mu, logvar)   # Reparameterize
        recon = self.decode(z)                # Decode
        return recon, mu, logvar
