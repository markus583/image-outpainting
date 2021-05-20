from loader import *
from architectures import SimpleCNN
from utils import load_config, load_pkl, save_pkl, plot
from scoring import scoring


def main(model_path: Union[str, Path], samples_path: Union[str, Path],
         config_path: Union[str, Path], pkl_path: Union[str, Path]):
    config = load_config(config_path)

    network_spec = config['network_config']

    # setup model and load state
    model = SimpleCNN()
    model.to(device=config['device'])
    model.load_state_dict(torch.load(model_path))

    # get samples from file
    input_arrays, known_arrays, borders_x, borders_y, ids = load_pkl(samples_path).values()

    predictions, masks = [], []
    model.eval()
    with torch.no_grad():
        # go through all samples, one by one
        for i, (input_array, known_array, border_x, border_y, ID) in enumerate(
                zip(input_arrays, known_arrays, borders_x, borders_y, ids)):
            # normalize sample
            transforms = TF.Compose([
                TF.ToTensor(),
                TF.Normalize([0.4848], [0.1883])  # TODO: compute actual values of TRAIN Set
            ])

            _input = transforms(input_array)  # normalized, but borders too! --> borders != 0
            masked_input = np.where(known_array, _input.detach().cpu().numpy(), 0)  # now borders are also 0
            masked_input = torch.from_numpy(masked_input).cuda()

            # get outputs
            output = model(masked_input.unsqueeze(0))
            prediction = output[0, 0][~known_array.astype(np.bool)]  # and border of outputs
            prediction = (prediction * 0.1883 + 0.4848) * 255  # un-normalize border/target outputs
            # append to list of predictions
            predictions.append((prediction.detach().cpu().numpy().astype(np.uint8)))
            # masks.append(bool_mask.detach().cpu().numpy())

    # save list containing all predictions as pkl file
    save_pkl(pkl_path, predictions)
    # plot(images, predictions, masks, SummaryWriter(log_dir=str(tb_path)))  # TODO: plot, sometimes at least
    true_target_path = r'C:\Users\Markus\Google Drive\linz\Subjects\Programming in Python\Programming in Python ' \
                       r'2\Assignment ' \
                       r'02\supplements_ex5\project\v2\python2-project\example_submission_perfect.pkl'
    print(scoring(pkl_path, true_target_path))


if __name__ == '__main__':
    model_path = r'C:\Users\Markus\Desktop\results\models\model_20_20210519-145431.pt'
    samples_path = r'C:\Users\Markus\Google Drive\linz\Subjects\Programming in Python\Programming in Python ' \
                   r'2\Assignment 02\supplements_ex5\project\v2\python2-project\example_testset.pkl '
    config_path = r'C:\Users\Markus\Google Drive\linz\Subjects\Programming in Python\Programming in Python ' \
                  r'2\Assignment 02\supplements_ex5\project\v2\python2-project\working_config.json '
    save_pkl_path = r'C:\Users\Markus\Desktop\results\save.pkl'
    main(model_path, samples_path, config_path, save_pkl_path)
