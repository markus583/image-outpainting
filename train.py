import os
from datetime import datetime
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_lr_finder import LRFinder
from torchinfo import summary

import eval
from architectures import *
from loader import PreprocessedImageDataset, train_test_split, image_collate_fn
from utils import load_config, plot

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def _plot_samples(
        epoch: int,
        model: torch.nn.Module,
        sample_batch,
        sample_mask,
        sample_targets,
        writer: SummaryWriter,
        path,
        show_all: bool = True,
        concat=True,
        stack=False,
):
    print("Plotting samples...")
    model.eval()
    sample_batch = sample_batch.to("cuda:0")
    sample_mask = sample_mask.to("cuda:0")
    if concat:
        concat_input = torch.cat((sample_batch, ~sample_mask), dim=1)
    elif stack:
        concat_input = torch.cat((sample_batch,) * 3, dim=1)
    else:
        concat_input = sample_batch
    output = model(concat_input)
    if len(output) == 1 or len(output) == 2:  # get proper output from PyTorch model zoo models
        output = output["out"]
    #  sample_prediction = [output[i, 0][sample_mask[i, 0]] for i in range(len(output))]
    preds = torch.zeros_like(sample_mask).type(torch.float32).to("cpu")  # setup tensor to store masked outputs
    output_and_input = sample_batch.clone().cpu()  # setup tensor to store combined inputs + outputs
    for i, sample in enumerate(output_and_input):
        # Get outputs, but without input values
        output_masked = np.where(sample_mask[i][0].cpu().numpy(), output[i][0].detach().cpu().numpy(), 0)
        output_only = output[i][0].detach().cpu().numpy()
        output_and_input[i, 0] = torch.from_numpy(output_masked).cpu()
        output_and_input[i, 0] += np.where(
            ~sample_mask[i][0].cpu().numpy(), sample_batch[i][0].cpu().numpy(), 0
        )  # add input to output
        # either show entire CNN output or only border values
        if show_all:
            preds[i][0] += output_only
        else:
            preds[i][0] += output_masked  # only add output values
    target_plotted = sample_mask.clone().type(torch.float32)

    # reconstruct target array
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
        path,
    )
    print("plots done!")


def setup_out_path(results_path):
    results_path = Path(results_path)  # general path/root
    results_path.mkdir(exist_ok=True)
    tensorboard_path = results_path / "tensorboard"  # folder for tensorboard file
    tensorboard_path.mkdir(exist_ok=True)

    model_path = results_path / "models"  # folder for models
    model_path.mkdir(exist_ok=True)

    return results_path, tensorboard_path, model_path


def main(
        dataset_path: Union[str, Path],
        config_path: Union[str, Path],
        results_path: Union[str, Path],
        epoch_no_change_break: int = 10,
        find_lr: bool = False,
        dataset_repeats: int = 3,
):
    # take care of possible KeyboardInterrupt
    try:
        # setup correct paths and tensorboard SummaryWriter
        results_path, tensorboard_path, model_path = setup_out_path(results_path)
        writer = SummaryWriter(log_dir=str(tensorboard_path))

        config = load_config(config_path)
        # save model config to csv file in result_path
        log_dict = {
            "dataset_path": dataset_path,
            "config_path": config_path,
            "result_path": results_path,
            "config": config,
            "network_config": config["network_config"],
        }
        log_df = pd.DataFrame(data=log_dict)
        log_df.to_csv(Path(results_path / "config.csv"))

        # get params
        network_spec = config["network_config"]
        n_workers = config["n_workers"]
        batch_size = config["batch_size"]
        n_epochs = config["n_epochs"]
        device = torch.device(config["device"])
        plotting_interval = config["plotting_interval"]
        LR = config["learning_rate"]

        # get dataset
        ds = PreprocessedImageDataset(dataset_path, uses=dataset_repeats)

        # get loaders
        # if no values are specified to train_test_split, then train is data_part_1-6
        # remainder is evenly split among test and valid set
        train, val, test = train_test_split(ds)
        train = DataLoader(
            train, batch_size=batch_size, num_workers=n_workers, collate_fn=image_collate_fn, shuffle=True
        )
        val = DataLoader(val, batch_size=batch_size, num_workers=n_workers, collate_fn=image_collate_fn, shuffle=False)
        test = DataLoader(
            test, batch_size=batch_size, num_workers=n_workers, collate_fn=image_collate_fn, shuffle=False
        )

        # get model
        model = SimpleCNN(
            n_hidden_layers=network_spec["n_hidden_layers"],
            n_kernels=network_spec["n_kernels"],
            kernel_size=network_spec["kernel_size"],
        )

        # model = pretrained_resnet().get()

        model.to(device=device)
        # if fine-tuning:
        # model.load_state_dict(torch.load(model_path_trained))
        concat = True  # use to control whether mask is fed into CNN
        stack = False  # controls whether grayscale channel is used 3 times --> pre-trained models
        print(summary(model))
        # optimizer
        optimizer = torch.optim.AdamW(params=model.parameters())

        # find best learning rate
        if find_lr:
            lr_finder = LRFinder(model, optimizer, torch.nn.CrossEntropyLoss(), device="cuda:0")
            lr_finder.range_test(train, end_lr=10, num_iter=1000)
            lr_finder.plot()
            plt.savefig("LRvsLoss.png")
            plt.close()
        # scheduler
        # scheduler = OneCycleLR(optimizer, max_lr=0.1, epochs=n_epochs, steps_per_epoch=len(train))

        # set up batches for tensorboard val plots
        sample_input, sample_mask, sample_targets = next(iter(val))

        # initialize -inf value as starting valid accuracy
        best_val_loss = torch.tensor(float("inf"))
        epoch_no_change = 0
        print("Start Training: ")
        for epoch in range(n_epochs):
            print(f"Epoch: {epoch}")

            train_loss = eval.train_eval(train, model, optimizer, concat=concat, stack=stack)
            print(f"train loss: {train_loss}")
            writer.add_scalar(tag="train/loss", scalar_value=train_loss, global_step=epoch)

            val_loss = eval.test_eval(val, model, concat=concat, stack=stack)
            print(f"valid loss: {val_loss}")
            writer.add_scalar(tag="valid/loss", scalar_value=val_loss, global_step=epoch)

            # plot every x times and in last epoch
            if epoch % plotting_interval == 0 or epoch == n_epochs - 1:
                _plot_samples(
                    epoch, model, sample_input, sample_mask, sample_targets, writer, results, concat=concat, stack=stack
                )

            # Add weights as arrays to tensorboard
            for i, param in enumerate(model.parameters()):
                writer.add_histogram(tag=f"training/param_{i}", values=param.cpu(), global_step=epoch)
            # Add gradients as arrays to tensorboard
            for i, param in enumerate(model.parameters()):
                writer.add_histogram(tag=f"training/gradients_{i}", values=param.grad.cpu(), global_step=epoch)

            print(f"Logs written into tensorboard: epoch: {epoch}")

            # save current best model if it has lowest validation loss so far
            if val_loss < best_val_loss:
                # current best model
                print("Saving current best model...")
                timestamp_best = datetime.now().strftime("%Y%m%d-%H%M%S")
                torch.save(
                    model.state_dict(), model_path / f"model_best_{epoch}_{timestamp_best}_{np.round(val_loss, 3)}.pt"
                )
                best_val_loss = val_loss
                epoch_no_change = 0
            else:
                epoch_no_change += 1
            # Stop the model from training further if it hasn't improved for epoch_no_change consecutive epochs
            if epoch_no_change > epoch_no_change_break:
                break

        print("Final model evaluation...")
        test_loss = eval.test_eval(test, model, concat=concat, stack=stack)
        val_loss = eval.test_eval(val, model, concat=concat, stack=stack)

        print(f"Final test loss: {test_loss}, Final valid loss: {val_loss}")
        print("Finished training.")

        # save final model
        timestamp_end = datetime.now().strftime("%Y%m%d-%H%M%S")
        torch.save(
            model.state_dict(), model_path / f"model_final_{n_epochs}_{timestamp_end}_{np.round(val_loss, 3)}.pt"
        )

        # save config + final values
        log_dict["final_test_loss"] = test_loss
        log_dict["final_valid_loss"] = val_loss
        log_df_final = pd.DataFrame(data=log_dict)
        log_df_final.to_csv(Path(results_path / "config_final.csv"))
    except (KeyboardInterrupt, IndexError):
        print("Finished training process due to keyboard interrupt.")
        # save last model state
        timestamp_end = datetime.now().strftime("%Y%m%d-%H%M%S")
        torch.save(
            model.state_dict(), model_path / f"model_interrupt_{n_epochs}_{timestamp_end}_{np.round(val_loss, 3)}.pt"
        )


if __name__ == "__main__":
    # setup own folder for each experiment
    timestamp_start = datetime.now().strftime("%Y%m%d-%H%M%S")
    results = r"C:\Users\Markus\Desktop\results\\"
    results += f"experiment_{timestamp_start}"
    config = (
        r"C:\Users\Markus\Google Drive\linz\Subjects\Programming in Python\Programming in Python 2\Assignment "
        r"02\supplements_ex5\project\v2\python2-project\working_config.json "
    )
    dataset = r"C:\Users\Markus\AI\dataset\dataset"

    main(dataset, config, results)

    # tensorboard --logdir tensorboard/ --port=6063
