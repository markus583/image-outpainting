import shutil
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Union, Tuple
import eval as tte
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from loader import PreprocessedImageDataset, crop, train_test_split, image_collate_fn
from architectures import SimpleCNN
from utils import load_config, plot
import numpy as np


# current best score: -778.8: bad config, only 5 epochs, data_part_1, leakage
# TODO: tensorboard
# TODO: better logging - csv of preds, losses
# TODO: un normalize where needed. Done?
# TODO: testing
# TODO: fix data leakage
# TODO: feed additional input into NN
# TODO: data augmentation
# TODO: save best model, not only last model
# TODO: save model if keyboard interrupt/other error
# TODO: use AdamW, change hyperparams?

def _plot_samples(epoch: int, model: torch.nn.Module, sample_batch, sample_mask, sample_targets,
                  writer: SummaryWriter, path):
    print('Plotting samples...')
    model.eval()
    sample_batch = sample_batch.to('cuda:0')
    output = model(sample_batch)
    #  sample_prediction = [output[i, 0][sample_mask[i, 0]] for i in range(len(output))]
    preds = torch.zeros_like(sample_mask).type(torch.float32)  # setup tensor to store masked outputs
    output_and_input = sample_batch.clone().cpu()  # setup tensor to store combined inputs + outputs
    for i, sample in enumerate(output_and_input):
        # Get outputs, but without input values
        output_masked = np.where(sample_mask[i][0], output[i][0].detach().cpu().numpy(), 0)
        output_and_input[i, 0] = torch.from_numpy(output_masked).cpu()
        output_and_input[i, 0] += np.where(~sample_mask[i][0].cpu().numpy(), sample_batch[i][0].cpu().numpy(),
                                           0)  # add input to output
        preds[i][0] += output_masked  # only add output values
    target_plotted = sample_mask.clone().type(torch.float32)
    for i, sample in enumerate(target_plotted):
        pos = 0
        for r, row in enumerate(sample[0]):
            for c, column in enumerate(row):
                if column:
                    if not pos >= sample_targets[i].shape.numel():
                        sample[0, r, c] = sample_targets[i][pos]
                        pos += 1

    plot(
        sample_batch[:, 0].cpu(),
        target_plotted[:, 0].cpu(),
        preds[:, 0].cpu(),
        output_and_input[:, 0].cpu(),
        writer,
        epoch,
        path
    )


def _setup_out_path(result_path: Union[str, Path]) -> Tuple[Path, Path, Path]:
    result_path = Path(result_path)
    result_path.mkdir(exist_ok=True)

    tb_path = result_path / 'tensorboard'
    # shutil.rmtree(tb_path, ignore_errors=True)
    tb_path.mkdir(exist_ok=True)

    model_path = result_path / 'models'
    model_path.mkdir(exist_ok=True)

    return result_path, tb_path, model_path


def main(dataset_path: Union[str, Path], config_path: Union[str, Path],
         result_path: Union[str, Path]):
    # paths and summary writer
    result_path, tb_path, model_path = _setup_out_path(result_path)
    writer = SummaryWriter(log_dir=str(tb_path))

    config = load_config(config_path)
    network_spec = config['network_config']

    # params
    n_workers = config['n_workers']
    batch_size = config['batch_size']
    n_epochs = config['n_epochs']
    device = torch.device(config['device'])
    plotting_interval = config['plotting_interval']
    # dataset
    ds = PreprocessedImageDataset(dataset_path)

    # loaders
    train, val, test = train_test_split(ds, train=0.7, val=0.15, test=0.15)
    train = DataLoader(train, batch_size=batch_size, num_workers=n_workers,
                       collate_fn=image_collate_fn
                       )
    val = DataLoader(val, batch_size=batch_size, num_workers=n_workers,
                     collate_fn=image_collate_fn
                     )
    test = DataLoader(test, batch_size=batch_size, num_workers=n_workers,
                      collate_fn=image_collate_fn
                      )

    # model
    model = SimpleCNN()
    model.to(device=device)

    # optimizer
    optimizer = torch.optim.Adam(params=model.parameters())

    # tensorboard visuals
    sample_input, sample_mask, sample_targets = next(iter(val))

    print('Start Training: ')
    for epoch in range(n_epochs):
        print(f'Epoch: {epoch}')

        train_loss = tte.train_eval(train, model, optimizer)
        print(f'train/loss: {train_loss}')
        writer.add_scalar(tag='train/loss', scalar_value=train_loss, global_step=epoch)

        val_loss = tte.test_eval(val, model)
        print(f'val/loss: {val_loss}')
        writer.add_scalar(tag='val/loss', scalar_value=val_loss, global_step=epoch)

        # plot every x times and after last epoch
        if epoch % plotting_interval == 0 or epoch == n_epochs-1:
            _plot_samples(epoch, model, sample_input, sample_mask, sample_targets, writer, results)

    print('Best model evaluation...')
    test_loss = tte.test_eval(test, model)
    val_loss = tte.test_eval(val, model)

    print(test_loss, val_loss)
    print('Finished training process.')

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    torch.save(model.state_dict(), model_path / f'model_{n_epochs}_{timestamp}.pt')


if __name__ == '__main__':
    results = r'C:\Users\Markus\Desktop\results'
    main(r'C:\Users\Markus\AI\dataset\dataset\data_part_1',
         r'C:\Users\Markus\Google Drive\linz\Subjects\Programming in Python\Programming in Python 2\Assignment '
         r'02\supplements_ex5\project\v2\python2-project\working_config.json',
         results)
