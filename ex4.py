"""
Author: Markus Frohmann
Matr.Nr.: k12005604
Exercise 4
"""

import numpy as np


def ex4(image_array: np.ndarray, border_x: tuple, border_y: tuple):
    if not isinstance(image_array, np.ndarray):
        raise NotImplementedError('Image is not a Numpy Array!')
    elif image_array.ndim != 2:
        raise NotImplementedError('Image is not a 2D array!')
    """
    if not (isinstance(border_x[0], int) and isinstance(border_x[1], int) and
            isinstance(border_y[0], int) and isinstance(border_y[1], int)):

        raise ValueError('One of the border values is not an integer!')
    """
    if border_x[0] < 1 or border_x[1] < 1 or border_y[0] < 1 or border_y[1] < 1:
        raise ValueError('One of the border values is smaller than 1!')

    input_array = image_array.copy()  # create copy of image
    # set values out of border to 0
    input_array[:border_x[0], :] = 0
    input_array[-border_x[1]:, :] = 0
    input_array[:, :border_y[0]] = 0
    input_array[:, -border_y[1]:] = 0

    known_array = input_array.copy()  # create copy of image with black borders
    # Set remaining values, i.e. those inside of borders, to 1
    known_array[border_x[0]:(known_array.shape[0] - border_x[1]), border_y[0]:(known_array.shape[1] - border_y[1])] = True

    # Create mask of same shape as input array, but without black pixels
    known = np.empty((image_array.shape[0] - sum(border_x), image_array.shape[1] - sum(border_y)))
    if len(known[0]) < 16 or len(known[1]) < 16:
        raise NotImplementedError('Remaining image is too small!')

    # create array where 1's == border, else 0
    boolean_border = np.invert(np.array(input_array, dtype=np.bool))
    # if any value inside the border is 0 (input pixel value in border is 0)
    boolean_border[border_x[0]:(known_array.shape[0] - border_x[1]),
        border_y[0]:(known_array.shape[1] - border_y[1])] = False
    target_array = image_array[boolean_border]
    # targets =
    return input_array, boolean_border, target_array
