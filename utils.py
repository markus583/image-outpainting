import os
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch
import json
import pickle
from pathlib import Path
from typing import Union, Any
from torch.utils.tensorboard import SummaryWriter


def plot(inputs, targets, predictions, combined, writer, epoch, path, dpi=300):
    """Plotting the inputs, targets, and predictions to file `path`"""
    plot_path = os.path.join(path, 'plots')
    os.makedirs(plot_path, exist_ok=True)
    fig, ax = plt.subplots(2, 2)

    for i in range(len(inputs)):
        # get min and max values of input image s.t. all images are plotted in same gray color
        vmax = inputs[i].max()
        vmin = inputs[i].min()
        ax[0, 0].clear()
        ax[0, 0].set_title('input')
        ax[0, 0].imshow(inputs[i], cmap=plt.cm.gray, interpolation='none', vmin=vmin, vmax=vmax)
        ax[0, 0].set_axis_off()
        ax[0, 1].clear()
        ax[0, 1].set_title('target')
        ax[0, 1].imshow(targets[i], cmap=plt.cm.gray, interpolation='none', vmin=vmin, vmax=vmax)
        ax[0, 1].set_axis_off()
        ax[1, 1].clear()
        ax[1, 1].set_title('prediction')
        ax[1, 1].imshow(predictions[i], cmap=plt.cm.gray, interpolation='none', vmin=vmin, vmax=vmax)
        ax[1, 1].set_axis_off()
        ax[1, 0].clear()
        ax[1, 0].set_title('input + prediction')
        ax[1, 0].imshow(combined[i], cmap=plt.cm.gray, interpolation='none', vmin=vmin, vmax=vmax)
        ax[1, 0].set_axis_off()
        fig.suptitle(f'Epoch: {epoch}, Image Nr.: {i}')
        fig.tight_layout()
        fig.savefig(os.path.join(plot_path, f"{epoch:07d}_{i:02d}.png"), dpi=dpi)
        writer.add_figure(tag=f'train/samples_{i}', figure=fig, global_step=epoch)
    del fig


def Normalize(ds):
    """
    :param ds: Dataset for which mean, std are to be computed.
    :return: mean, std of all images in ds
    """
    loader = DataLoader(ds,
                        batch_size=10,
                        num_workers=0,
                        shuffle=False)

    mean = 0.
    std = 0.
    for images in loader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    return mean, std


def load_config(path: Union[Path, str]):
    with open(Path(path), 'r') as f:
        return json.load(f)


def load_pkl(path: Union[str, Path]) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pkl(path: Union[str, Path], obj: Any):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)