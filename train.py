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
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Third-party
from utils.callbacks import CallBack


class Trainer():

    def __init__(
            self,
            model: nn.Module,
            criteria,
            optimizer: optim.Optimizer,
            device: torch.device = torch.device("cuda")
        ) -> 'Trainer':
        """
            Trainer

            Parameters:
                model (nn.Module): Model to train
                criteria: Loss function
                optimizer (optim.Optimizer): Optimization strategy
                device (torch.device): Device to mount the training on
        """
        self.model = model
        self.criteria = criteria
        self.optimizer = optimizer
        self.device = device


    def validate(self, dataloader: DataLoader) -> torch.Tensor:
        """
            Evaluates the model on the given batch

            Parameters:
                dataloader (DataLoader): Valdiation data
            Returns:
                (torch.Tensor): Mean batch loss
        """
        loss = kl = mse = 0.0
        self.model.eval()       # Set evaluation mode

        print("Validating", end = "\t")
        with torch.no_grad():
            for _, batch in enumerate(dataloader):
                # Read Batch
                images = batch.to(self.device)

                outputs, mean, logvar = self.model(images)
                # Calculate loss
                _kl, _mse, _loss = self.criteria(images, outputs, logvar, mean)
                kl += _kl
                mse += _mse
                loss += _loss
                print("=", end = "")

        print(">", end = " ")
        return kl / len(dataloader), mse / len(dataloader), loss / len(dataloader)


    def train(self, dataloader: DataLoader) -> torch.Tensor:
        """
            Trains the model on the given batch

            Parameters:
                dataloader (DataLoader): Training data
            Returns:
                (torch.Tensor): Mean batch loss
        """
        loss = kl = mse = 0.0
        self.model.train()      # Set training mode

        print("Training", end = "\t")
        for _, batch in enumerate(dataloader):
            # Read Batch
            images = batch.to(self.device)

            outputs, mean, logvar = self.model(images)
            _kl, _mse, _loss = self.criteria(images, outputs, mean, logvar)
            self.optimizer.zero_grad()  # Reset Gradients
            _loss.backward()            # Propagate
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()       # Update Parameters
            kl += _kl
            mse += _mse
            loss += _loss
            print("=", end = "")

        print(">", end = " ")
        return kl / len(dataloader), mse / len(dataloader), loss / len(dataloader)


    def run(
        self,
        start_epoch: int,
        end_epoch: int,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        tb_writer: SummaryWriter = None,
        save_plot: bool = True,
        identifier: str = "",
    ) -> None:
        """
            Main Loop

            Parameters:
                start_epoch (int): Starting Epoch
                end_epoch (int): Ending Epoch
            Returns:
                None
        """
        callback = CallBack(train_loader, valid_loader)

        for epoch in range(start_epoch, end_epoch):
            print(f"\nEPOCH {epoch})")

            # Train
            train_kl, train_mse, train_loss = self.train(train_loader)
            print(f"KL: {round(train_kl.item())}\tMSE: {round(train_mse.item())}\tLoss: {round(train_loss.item())}")
            if epoch % 10 == 0:
                callback.on_train_end(self.model, f"results/images/{identifier}/train_recon_{epoch}.jpg" if save_plot else None)

            # Validate
            valid_kl, valid_mse, valid_loss = self.validate(valid_loader)
            print(f"KL: {round(valid_kl.item())}\tMSE: {round(valid_mse.item())}\tLoss: {round(valid_loss.item())}")
            if epoch % 20 == 0:
                callback.on_valid_end(self.model, f"results/images/{identifier}/valid_random_recon_{epoch}.jpg" if save_plot else None)

            # Tensorboard
            if tb_writer:
                for name, metric in [
                    # KL
                    ('kl/train', train_kl), ('kl/valid', valid_kl),
                    # MSE
                    ('mse/train', train_mse), ('mse/valid', valid_mse),
                    # Total
                    ('loss/train', train_loss), ('loss/valid', valid_loss),
                ]: tb_writer.add_scalar(name, metric.item(), epoch)

            if epoch % 50 == 0:
                callback.on_epoch_end(self.model, f"results/weights/{epoch}.pth")
