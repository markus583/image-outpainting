# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich
Contact -- widrich@ml.jku.at
Date -- 01.10.2020

###############################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This  material,  no  matter  whether  in  printed  or  electronic  form,
may  be  used  for personal  and non-commercial educational use only.
Any reproduction of this manuscript, no matter whether as a whole or in parts,
no matter whether in printed or in electronic form, requires explicit prior
acceptance of the authors.

###############################################################################

"""

import dill as pkl
import numpy as np


def mse(target_array, prediction_array):
    if prediction_array.shape != target_array.shape:
        raise IndexError(f"Target shape is {target_array.shape} but prediction shape is {prediction_array.shape}")
    prediction_array, target_array = np.asarray(prediction_array, np.float64), np.asarray(target_array, np.float64)
    return np.mean((prediction_array - target_array) ** 2)


def scoring(prediction_file: str, target_file: str):
    """Computes the mean mse loss on two lists of numpy arrays stored in pickle files prediction_file and targets_file

    Computation of mean mse loss, as used in the challenge for exercise 5. See files "example_testset.pkl" and
    "example_targets.pkl" for an example test set and example targets, respectively. The real test set (without targets)
    will be available via the challenge server.

    Parameters
    ----------
    prediction_file: str
        File path of prediction file. Has to be a pickle file (or dill file) and contain a list of numpy arrays of dtype
        uint8, as specified in assignment sheet 2.
    target_file: str
        File path of target file. Has to be a pickle file (or dill file) and contain a list of numpy arrays of dtype
        uint8, as specified in assignment sheet 2. This file will not be available for the challenge. See file
        "example_targets.pkl" for an example.
    """
    # Load predictions
    with open(prediction_file, "rb") as pfh:
        predictions = pkl.load(pfh)
    if not isinstance(predictions, list):
        raise TypeError(
            f"Expected a list of numpy arrays as pickle file. "
            f"Got {type(predictions)} object in pickle file instead."
        )
    if not all([isinstance(prediction, np.ndarray) and np.uint8 == prediction.dtype for prediction in predictions]):
        raise TypeError("List of predictions contains elements which are not numpy arrays of dtype uint8")

    # Load targets
    with open(target_file, "rb") as tfh:
        targets = pkl.load(tfh)
    if len(targets) != len(predictions):
        raise IndexError(
            f"list of targets has {len(targets)} elements "
            f"but list of submitted predictions has {len(predictions)} elements."
        )

    # Compute MSE for each sample
    mses = [mse(target, prediction) for target, prediction in zip(targets, predictions)]

    return -np.mean(mses)


if __name__ == "__main__":
    submission_path = r"C:\Users\Markus\Desktop\results\save.pkl"
    target_path = (
        r"C:\Users\Markus\Google Drive\linz\Subjects\Programming in Python\Programming in Python "
        r"2\Assignment "
        r"02\supplements_ex5\project\v2\python2-project\example_submission_perfect.pkl"
    )
    # raise FileNotFoundError("Scoring on test set will be possible with 24th of June, 2020 (see exercise 5).")
    mse_loss = scoring(prediction_file=submission_path, target_file=target_path)
    print(f"Score: {mse_loss}")
