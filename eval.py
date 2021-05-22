import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def _eval(loader: DataLoader, model: nn.Module,
          optimizer=None, scheduler=False,
          concat=True, stack=False) -> float:
    """
    General purpose function used to train model and check its performance on a given DataLoader.
    :param loader: DataLoader
    :param model: PyTorch model - CNN
    :param optimizer: PyTorch Optimizer
    :param scheduler: optional, also during training. Schedules LR.
    :return: total loss, normalized by number and size of mini batches.
    """
    is_train = optimizer is not None  # only train if optimizer is specified
    device = next(model.parameters()).device

    mse_loss = nn.MSELoss()
    total_loss = 0
    for _input, mask, targets in tqdm(loader, total=len(loader)):
        _input = _input.to(device=device)
        mask = mask.to(device=device)
        if concat:
            concat_input = torch.cat((_input, ~mask), dim=1)
        elif stack:
            concat_input = torch.cat((_input, )*3, dim=1)
        else:
            concat_input = _input
        compute_grad = torch.enable_grad() if is_train else torch.no_grad()
        with compute_grad:
            output = model(concat_input)  # get model output
            # get output from borders only
            if len(output) == 1 or len(output) == 2:  # get proper output from PyTorch model zoo models
                output = output['out']
            predictions = [output[i, 0][mask[i, 0]] for i in range(len(output))]
            # compute loss
            losses = torch.stack(
                [mse_loss(prediction, target.to(device=device).view(-1))
                 for prediction, target in zip(predictions, targets)])
            loss = losses.sum()  # get scalar

            # change weights
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # change LR
                if scheduler:
                    scheduler.step()
            total_loss += loss.detach().cpu().item()  # finally, accumulate losses
    return total_loss / (len(loader) * loader.batch_size)


def train_eval(train_loader: DataLoader, model: nn.Module, optimizer, scheduler=None, concat=True, stack=False) -> float:
    """
    Function used to train model and check model performance on train set.
    :param stack: Whether to stack grayscale 1D-input to grayscale 3D-input.
    :param concat: Whether to concatenate mask with input or not.
    :param train_loader: DataLoader for Training Set
    :param model: PyTorch model - CNN
    :param optimizer: PyTorch Optimizer
    :param scheduler: optional, also during training. Schedules LR.
    :return: total loss on training set.
    """
    assert optimizer is not None  # optimizer is needed
    model.train()
    return _eval(train_loader, model, optimizer, scheduler, concat=concat, stack=stack)


def test_eval(test_loader: DataLoader, model: nn.Module, concat=True, stack=False):
    """
    Function used to check model performance on valid/test set.
    :param test_loader: DataLoader for valid/test set
    :param model: PyTorch model - CNN
    :return: total loss on training/evaluation set.
    """
    model.eval()
    return _eval(test_loader, model, concat=concat, stack=stack)
