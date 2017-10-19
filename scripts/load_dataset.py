import os

import matplotlib.pyplot as plt
import numpy as np


def load_dataset(path):
    """
    Loads dataset as numpy objects. The data will be shuffled to avoid
    pre-existing ordering.

    Arguments:
    path: path of the file from current directory.

    Returns:
    X -- input data, shape: number of features x number of examples.
    Y -- label vector, shape: 1 x number of examples.
    """
    # set up the path
    os.chdir(path)

    # get all file names to iterate over all of them
    image_list_names = os.listdir()[1:]
    m = len(image_list_names)

    # loading images
    X = []

    for img in image_list_names:
        if img.startswith("."):
            m -= 1
        
        else:
            temp = np.array(plt.imread(img))
            X.append(temp)
            
    # convert to numpy array
    X = np.array(X)

    # reshape X
    X = X.reshape(m, -1).T

    # Derive true label vector
    Y = np.zeros((1, m))

    for i, img in enumerate(image_list_names):
        if img.startswith("cat"):
            Y[:, i] = 1
            
        elif img.startswith("dog"):
            Y[:, i] = 0

    # shuffle the dataset
    permutation = np.random.permutation(m)
    X = X[:, permutation]
    Y = Y[:, permutation]

    return X, Y