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
from typing import Tuple, List, Optional

# Third-party
from .layers import Encoder, Decoder, OutLayer


class VAE(nn.Module):

    def __init__(
            self,
            in_channels: int,
            latent_dim: int,
            hidden_dims: List[int] = [32, 64, 128, 256, 512],
            device: torch.device = torch.device('cpu'),
            has_skip: bool = True
        ) -> 'VAE':
        super(VAE, self).__init__()
        self.device = device
        self.has_skip = has_skip
        self.latent_dim = latent_dim

        modules = []

        # Encoder
        for h_dim in hidden_dims:
            modules.append(Encoder(in_channels, h_dim))
            in_channels = h_dim
        self.encoder_1 = nn.Sequential(*modules)
        self.encoder_2 = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(2048, 2048),
            # nn.LeakyReLU(inplace=False),
        )

        # Latent Space
        self.mean_layer = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # Decoder
        modules = []
        self.decoder_1 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[-1] * 4),
            nn.LeakyReLU(inplace=False),
            # nn.Linear(hidden_dims[-1] * 2, hidden_dims[-1] * 4),
            # nn.LeakyReLU(inplace=False),
        )

        hidden_dims.reverse()
        if has_skip:
            hidden_dims[0] *= 2
        for i in range(len(hidden_dims) - 1):
            modules.append(Decoder(hidden_dims[i], hidden_dims[i + 1]))

        self.decoder_2 = nn.Sequential(*modules)
        self.decoder_out = OutLayer(hidden_dims[-1], hidden_dims[-1])


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
        feat = self.encoder_1(x)            # Encode
        x = self.encoder_2(feat)            # Encode
        mu = self.mean_layer(x)             # Mean
        logvar = self.logvar_layer(x)       # Logvar

        if self.has_skip:
            return mu, logvar, feat
        return mu, logvar


    def decode(self, x: torch.Tensor, feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
            Forward pass for the decoder

            Parameters:
                x (torch.torch.Tensor): Input
            Returns:
                (torch.torch.Tensor): Reconstructed image
        """
        x = self.decoder_1(x)
        x = x.view(-1, 512, 2, 2)   # Reshape

        if self.has_skip:
            x = torch.cat((x, feat), dim = 1)

        x = self.decoder_2(x)       # Decode
        x = self.decoder_out(x)     # Decoder output
        return x


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
            Forward pass for the VAE

            Parameters:
                x (torch.torch.Tensor): Input image
            Returns:
                (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Reconstructed image, mean and logvar
        """
        # Encode
        if self.has_skip:   mu, logvar, feat = self.encode(x)
        else:               mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)   # Reparameterize

        # Decode
        if self.has_skip:   recon = self.decode(z, feat)
        else:               self.decode(z, feat)

        return recon, mu, logvar
