import os
import glob
import numpy as np
import torch
import torch.utils.data
# from utils import plot
# from architectures import SimpleCNN
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import tqdm
from ex2 import ex2
from ex3 import ImageNormalizer
from ex4 import ex4
from skimage import io
from PIL import Image
from torch.utils.data import DataLoader, Subset, Dataset
from architectures import SimpleCNN
from utils import plot


BORDER_X = (5, 5)
BORDER_Y = (5, 5)

"""Main function that takes hyperparameters and performs training and evaluation of model"""
# Prepare a path to plot to
plotpath = os.path.join(os.getcwd(), 'plots')
os.makedirs(plotpath, exist_ok=True)


def image_collate_fn(image_batch: list):
    #
    # Handle sequences
    #
    # Get sequence entries, which are at index 0 in each sample tuple
    images = [sample for sample in image_batch]
    # Get the maximum sequence length in the current mini-batch
    max_X = np.max([image[0].shape[0] for image in images])
    max_Y = np.max([image[0].shape[1] for image in images])
    # Allocate a tensor that can fit all padded sequences
    n_feature_channels = 1  # Could be hard-coded to 3
    stacked_images = torch.zeros(size=(len(images), n_feature_channels,
                                       max_X, max_Y), dtype=torch.float32)
    # Write the sequences into the tensor stacked_sequences
    for i, image in enumerate(images):
        stacked_images[i, :] = torch.from_numpy(image[0])

    return stacked_images


# def get_targets()


input_path = r'C:\Users\Markus\AI\dataset\dataset\data_part_1'
# image_files = sorted(glob.glob(os.path.join(input_path, '**', '*.jpg'), recursive=True))
image_files = ImageNormalizer(r'C:\Users\Markus\AI\dataset\dataset\data_part_1\000')
images_mean, images_std = image_files.analyze_images()
images_mean = 120.5
images_std = 48.46
# image_files.get_images_data()


class ImageData(Dataset):
    def __init__(self, image_files):
        self.image_data = image_files

    def __len__(self):
        return len(self.image_data.file_paths_abs)

    def __getitem__(self, item):
        im_shape = 90
        resize_transforms = transforms.Compose([
            transforms.Resize(size=im_shape),
            transforms.CenterCrop(size=(im_shape, im_shape)),
        ])
        image = Image.open(self.image_data.file_paths_abs[item])
        image = resize_transforms(image)
        image -= self.image_data.mean
        image /= self.image_data.std
        input_array, known_array, target_array = ex4(image, BORDER_X, BORDER_Y)
        return input_array, known_array, target_array, item


dataset = ImageData(image_files)
print(image_files)
input = dataset[0]
print(input)

our_dataloader = DataLoader(dataset,  # we want to load our dataset
                            shuffle=False,  # shuffle the order of our samples
                            batch_size=4,  # stack 4 samples to a minibatch
                            num_workers=0,  # no background workers for now
                            collate_fn=image_collate_fn
                            )
for samples in our_dataloader:
    # print(samples.shape)
    pass
    # sample = samples[0]

trainingset = Subset(our_dataloader.dataset, indices=np.arange(int(len(our_dataloader) * (3 / 5))))
validationset = Subset(our_dataloader, indices=np.arange(int(len(our_dataloader) * (3 / 5)),
                                                         int(len(our_dataloader) * (4 / 5))))
testset = Subset(our_dataloader, indices=np.arange(int(len(our_dataloader) * (4 / 5)),
                                                   len(our_dataloader)))

trainloader = DataLoader(dataset=trainingset, batch_size=4, shuffle=True,
                         num_workers=0)
n_updates = 50
update = 0    # Create Network
net = SimpleCNN()
device = torch.device("cuda:0")
net.to(device)

# Get mse loss function
mse = torch.nn.MSELoss()

# Get adam optimizer
optimizer = torch.optim.Adam(net.parameters())

print_stats_at = 1e2  # print status to tensorboard every x updates
plot_at = 1e4  # plot every x updates
validate_at = 5e3  # evaluate model on validation set and check for new best model every x updates
best_validation_loss = np.inf  # best validation loss so far
while update < n_updates:
    for data in trainloader:
        # Get next samples in `trainloader_augmented`
        inputs, unknown, targets, ids = data
        inputs = inputs[:, None, :, :]
        unknown = unknown[:, None, :, :]
        inputs = inputs.to(device, dtype=torch.float32)
        targets = targets.to(device, dtype=torch.float32)
        unknown = unknown.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Get outputs for network
        outputs = net(inputs)
        outputs_1 = outputs[unknown].reshape(4, -1)
        # Calculate loss, do backward pass, and update weights
        loss = mse(outputs_1, targets)
        loss.backward()
        optimizer.step()

        # Print current status and score
        if update % print_stats_at == 0 and update > 0:
            writer.add_scalar(tag="training/loss",
                              scalar_value=loss.cpu(),
                              global_step=update)

        # Plot output
        if update == 20 or update == 45:
            plot(inputs.detach().cpu().numpy(), known.detach().cpu().numpy(), outputs.detach().cpu().numpy(),
                 plotpath, update)

        # Evaluate model on validation set
        if update % validate_at == 0 and update > 0:
            val_loss = evaluate_model(net, dataloader=valloader, device=device)
            writer.add_scalar(tag="validation/loss", scalar_value=val_loss.cpu(), global_step=update)
            # Add weights as arrays to tensorboard
            for i, param in enumerate(net.parameters()):
                writer.add_histogram(tag=f'validation/param_{i}', values=param.cpu(),
                                     global_step=update)
            # Add gradients as arrays to tensorboard
            for i, param in enumerate(net.parameters()):
                writer.add_histogram(tag=f'validation/gradients_{i}',
                                     values=param.grad.cpu(),
                                     global_step=update)
            # Save best model for early stopping
            if best_validation_loss > val_loss:
                best_validation_loss = val_loss
                torch.save(net, os.path.join(results_path, 'best_model.pt'))

        #update_progess_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
        #update_progess_bar.update()

        # Increment update counter, exit if maximum number of updates is reached
        update += 1
        if update >= n_updates:
            break

update_progess_bar.close()
print('Finished Training!')

# Load best model and compute score on test set
print(f"Computing scores for best model")
net = torch.load(os.path.join(results_path, 'best_model.pt'))
test_loss = evaluate_model(net, dataloader=testloader, device=device)
val_loss = evaluate_model(net, dataloader=valloader, device=device)
train_loss = evaluate_model(net, dataloader=trainloader, device=device)

print(f"Scores:")
print(f"test loss: {test_loss}")
print(f"validation loss: {val_loss}")
print(f"training loss: {train_loss}")

# Write result to file
with open(os.path.join(results_path, 'results.txt'), 'w') as fh:
    print(f"Scores:", file=fh)
    print(f"test loss: {test_loss}", file=fh)
    print(f"validation loss: {val_loss}", file=fh)
    print(f"training loss: {train_loss}", file=fh)
