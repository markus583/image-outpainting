from torch.utils.data import DataLoader, Subset, Dataset
from typing import Tuple, List, Union, Dict, NamedTuple, Callable
from pathlib import Path
import torch
from torchvision.transforms import transforms as TF
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
IM_SHAPE = 90


class PreprocessedImageDataset(Dataset):
    def __init__(self, root, uses: int = 1):
        super().__init__()

        self.uses = uses
        self.paths = sorted(self.image_paths(Path(root)))
        self.transforms = A.Compose([
            A.RandomResizedCrop(height=90, width=90),
            A.Flip(),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize([0.48645], [0.2054]),
            ToTensorV2()
        ])

    def __getitem__(self, item):
        # compute transforms of single image
        # returns tuple of tensors of shape (90, 90), (90, 90), 0-d tensor dependent on border size
        self.image = self.transforms(image=np.array(Image.open(self.paths[item])))
        self.input, self.known_mask, self.target = GetInOut(self.image['image'])
        return self.input, self.known_mask, self.target

    def __len__(self):
        return len(self.paths)

    def image_paths(self, root):
        """
        Recursively get all images in root as list
        """
        paths = list(root.rglob('*.jpg')) * self.uses
        return paths


def crop(image_array: torch.Tensor, border_x: tuple, border_y: tuple):
    """
    essentially ex4 without error handling and with tensors
    """
    input_array = image_array.clone()  # create copy of image
    # set values out of border to 0
    input_array[:, :border_x[0], :] = 0
    input_array[:, -border_x[1]:, :] = 0
    input_array[:, :, :border_y[0]] = 0
    input_array[:, :, -border_y[1]:] = 0

    known_array = input_array.clone()  # create copy of image with black borders
    # Set remaining values, i.e. those inside of borders, to 1
    known_array[border_x[0]:(known_array.shape[0] - border_x[1]),
    border_y[0]:(known_array.shape[1] - border_y[1])] = True

    # Create mask of same shape as input array, but without black pixels
    # known = np.empty((image_array.shape[0] - sum(border_x), image_array.shape[1] - sum(border_y)))

    # create array where 1's == border, else 0
    boolean_border = np.invert(np.array(input_array, dtype=np.bool))
    # if any value inside the border is 0 (input pixel value in border is 0)
    boolean_border[border_x[0]:(known_array.shape[0] - border_x[1]),
    border_y[0]:(known_array.shape[1] - border_y[1])] = False
    boolean_border = torch.from_numpy(boolean_border)  # convert to tensor
    target_array = image_array[boolean_border]  # now usable with tensor
    return input_array, boolean_border, target_array


def GetInOut(image, border_x=None, border_y=None):
    """
    Gets input, known mask, and outputs from preprocessed image.
    If no border is specified, randomly create border values --> Train
    If border is specified, use this border --> Test
    """
    if border_y is None:
        border_y = np.random.randint(5, 15, 2).astype(int)
    if border_x is None:
        border_x = np.random.randint(5, 15, 2).astype(int)
    return crop(image, border_x=border_x, border_y=border_y)


def train_test_split(dataset: Dataset, train_size: float = 0.7482291345857714,
                     val_size: float = 0.12588543270711428, test_size: float = 0.12588543270711428, seed: int = 0):
    torch.manual_seed(seed)
    # use indices from splits
    train = torch.utils.data.Subset(dataset, indices=np.arange(int(len(dataset) * train_size)))
    val = torch.utils.data.Subset(dataset, indices=np.arange(int(len(dataset) * train_size),
                                                             int(len(dataset) * (val_size + train_size))))
    test = torch.utils.data.Subset(dataset, indices=np.arange(int(len(dataset) * (train_size + val_size)),
                                                              int(len(dataset))))  # just use remainder of dataset
    return train, val, test


def image_collate_fn(image_batch: list, n_feature_channels: int = 1):
    #
    # Handle sequences
    #
    # Get sequence entries, which are at index 0 in each sample tuple
    images = [sample for sample in image_batch]
    # Get the maximum number of height and width of image
    max_X = np.max([image[0][0].shape[0] for image in images])
    max_Y = np.max([image[0][0].shape[1] for image in images])
    max_target = np.max([image[2].shape for image in images])
    # Allocate a tensor that can fit all padded sequences
    stacked_images_input = torch.zeros(size=(len(images), n_feature_channels,
                                             max_X, max_Y), dtype=torch.float32)
    stacked_images_mask = stacked_images_input.clone()
    # only if all targets should have equal size:
    # stacked_images_target = torch.zeros(size=(len(images), n_feature_channels, max_target), dtype=torch.float32)
    stacked_images_target = []  #
    # Write the sequences into the tensor stacked_sequences
    for i, image in enumerate(images):
        stacked_images_input[i, 0, :image[0][0][i].shape.numel(), :image[1][0][i].shape.numel()] = image[0]
        stacked_images_mask[i, 0, :image[1][0][i].shape.numel(), :image[1][0][i].shape.numel()] = image[1]
        # stacked_images_target[i, 0, :image[2].shape.numel()] = image[2]  # only if all targets should have equal size
        stacked_images_target.append(image[2])
    return stacked_images_input, stacked_images_mask.bool(), stacked_images_target
