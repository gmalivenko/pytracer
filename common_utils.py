import numpy as np


def normalize(x):
    """
    Normalization (x) / |x| function
    :param x: Vector to normalize
    :return: Normalized vector
    """
    return x / np.linalg.norm(x)


epsilon = 0.00001
