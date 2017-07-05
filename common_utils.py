import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing as mp

def normalize(x):
    return x / np.linalg.norm(x)

epsilon = 0.00001