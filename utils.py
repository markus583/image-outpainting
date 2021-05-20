import os
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import json
import pickle
from pathlib import Path
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch


def plot(inputs, targets, predictions, combined, writer, epoch, path, dpi=300):
    """Plotting the inputs, targets, and predictions to file `path`"""
    plot_path = os.path.join(path, 'plots')
    os.makedirs(plot_path, exist_ok=True)
    fig, ax = plt.subplots(2, 2)

    for i in range(len(inputs)):
        # get min and max values of input image s.t. all images are plotted in same gray color
        vmax = inputs[i].max()
        vmin = inputs[i].min()

        # create plots
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
        # save plots and add to tensorboard
        fig.savefig(os.path.join(plot_path, f"{epoch:07d}_{i:02d}.png"), dpi=dpi)
        writer.add_figure(tag=f'train/samples_{i}', figure=fig, global_step=epoch)
    del fig


def Normalize(ds):
    """
    Get mean, std of an image DataSet.
    :param ds: Dataset for which mean, std are to be computed.
    :return: mean, std of all images in ds
    """
    loader = DataLoader(ds,
                        batch_size=10,
                        num_workers=0,
                        shuffle=False)

    mean = 0.
    std = 0.
    for i, images in enumerate(loader):
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.type(torch.float32).mean(2).sum(0)
        std += images.type(torch.float32).std(2).sum(0)
        if i % 100 == 0:
            print(i)
    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    return mean, std


def load_config(path):
    with open(Path(path), 'r') as f:
        return json.load(f)


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pkl(path, file):
    with open(path, 'wb') as f:
        pickle.dump(file, f)


class ImageDataset(Dataset):
    def __init__(self, root, uses: int = 1):
        super().__init__()

        self.uses = uses
        self.paths = sorted(self.image_paths(Path(root)))
        self.transforms = A.Compose([
            A.transforms.Resize(height=90, width=90),
            A.CenterCrop(height=90, width=90),
            ToTensorV2()
        ])
        # com
        # pute transforms of single image
        # returns tuple of tensors of shape (90, 90), (90, 90), 0-d tensor dependent on border size

    def __getitem__(self, item):
        self.image = self.transforms(image=(np.array(Image.open(self.paths[item]))))
        return self.image['image']

    def __len__(self):
        return len(self.paths)

    def image_paths(self, root):
        """
        Recursively get all images in root as list
        """
        paths = list(root.rglob('*.jpg')) * self.uses
        return paths


#dataset = ImageDataset(r'C:\Users\Markus\AI\dataset\dataset')
#mean, std = Normalize(dataset)
#print(mean, std)
