import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import Union, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import transforms as TF
from torch.utils.tensorboard import SummaryWriter

from loader import *
from architectures import SimpleCNN
from utils import load_config, load_pkl, save_pkl


def main(model_path: Union[str, Path], samples_path: Union[str, Path],
         config_path: Union[str, Path], pkl_path: Union[str, Path]):
    config = load_config(config_path)

    network_spec = config['network_config']

    # model
    model = SimpleCNN()
    model.to(device=config['device'])
    model.load_state_dict(torch.load(model_path))

    # norm = ZeroOneScaler()
    input_arrays, known_arrays, borders_x, borders_y, ids = load_pkl(samples_path).values()

    predictions, masks = [], []
    model.eval()
    with torch.no_grad():
        # see custom_collate_fn
        for i, (input_array, known_array, border_x, border_y, ID) in enumerate(
                zip(input_arrays, known_arrays, borders_x, borders_y, ids)):
            transforms = TF.Compose([
                TF.ToTensor(),
                TF.Normalize([0.4848], [0.1883])  # TODO: compute actual values of TRAIN Set
            ])
            input = transforms(input_array)  # normalized, but borders too! --> borders != 0
            masked_input = np.where(known_array, inp.detach().cpu().numpy(), 0)  # so that borders are also 0
            masked_input = torch.from_numpy(masked_input).cuda()
            output = model(masked_input.unsqueeze(0))  # get outputs
            prediction = output[0, 0][~known_array.astype(np.bool)]  # and border of outputs
            prediction = (prediction * 0.1883 + 0.4848) * 255  # un-normalize border/target outputs
            # TODO: try/compare with sample submissions/scoring function. P1!
            predictions.append((prediction.detach().cpu().numpy().astype(np.uint8)))
            masks.append(bool_mask.detach().cpu().numpy())

    save_pkl(pkl_path, predictions)
    _plot_predicted(images, predictions, masks, SummaryWriter(log_dir=str(tb_path)))


if __name__ == '__main__':
    model_path = r'C:\Users\Markus\Desktop\results\models\model_5_20210518-201948.pt'
    samples_path = r'C:\Users\Markus\Google Drive\linz\Subjects\Programming in Python\Programming in Python ' \
                   r'2\Assignment 02\supplements_ex5\project\v2\example_testset.pkl '
    config_path = r'C:\Users\Markus\Google Drive\linz\Subjects\Programming in Python\Programming in Python ' \
                  r'2\Assignment 02\supplements_ex5\project\v2\working_config.json '
    save_path = r'C:\Users\Markus\Desktop\results'
    main(model_path, samples_path, config_path, save_path)
