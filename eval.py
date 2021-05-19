from typing import Optional

import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


def _eval(loader: DataLoader, model: nn.Module, optimizer: Optional[Optimizer] = None, scheduler=None) -> float:
    """
    General purpose evaluation function
    Parameters
    ----------
    loader : DataLoader
        data loader
    model : nn.Module
        CNN to evaluate
    optimizer : Optional[Optimizer]
        optimizer to update during training
    Returns
    -------
    loss : float
        total loss divided by len(loader)
    """
    is_train = optimizer is not None
    device = next(model.parameters()).device

    mse_loss = MSELoss()
    total_loss = 0
    for _input, mask, targets in tqdm(loader, total=len(loader)):
        _input = _input.to(device=device)

        compute_grad = torch.enable_grad() if is_train else torch.no_grad()
        with compute_grad:
            output = model(_input)
            predictions = [output[i, 0][mask[i, 0]] for i in range(len(output))]
            losses = torch.stack(
                [mse_loss(prediction, target.to(device=device).view(-1)) for
                 prediction, target in zip(predictions, targets)])
            loss = losses.sum()

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
            total_loss += loss.detach().cpu().item()
    return total_loss / len(loader)


def train_eval(train_loader: DataLoader, model: nn.Module, optimizer: Optimizer, scheduler) -> float:
    """
    Evaluate model during training
    Parameters
    ----------
    train_loader : DataLoader
        loader for train data
    model : nn.Module
        CNN to evaluate
    optimizer : Optimizer
        optimizer to update
    Returns
    -------
    loss : float
        total loss divided by len(train_loader)
    """
    assert optimizer is not None
    model.train()
    return _eval(train_loader, model, optimizer, scheduler)


def test_eval(test_loader: DataLoader, model: nn.Module):
    """
    Evaluate model during training
    Parameters
    ----------
    test_loader : DataLoader
        loader for validation or test data
    model : nn.Module
        CNN to evaluate
    Returns
    -------
    loss : float
        total loss divided by len(test_loader)
    """
    model.eval()
    return _eval(test_loader, model)