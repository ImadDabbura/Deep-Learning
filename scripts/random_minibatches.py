import numpy as np


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, shape: number of examples x number of features.
    Y -- "label" vector, shape: number of examples x 1.
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    np.random.seed(seed)
    m = X.shape[0]
    mini_batches = []

    # shuffle training set
    permutation = np.random.permutation(m)
    shuffle_X = X[permutation, :]
    shuffle_Y = Y[permutation, :]

    num_complete_minibatches = m // mini_batch_size

    for k in range(num_complete_minibatches):
        mini_batch_X = shuffle_X[k*mini_batch_size:(k + 1)*mini_batch_size, :]
        mini_batch_Y = shuffle_Y[k*mini_batch_size:(k + 1)*mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # check if there are some examples left if m % batch_size != 0
    if m % mini_batch_size != 0:
        mini_batch_X = shuffle_X[num_complete_minibatches*mini_batch_size:, :]
        mini_batch_Y = shuffle_Y[num_complete_minibatches*mini_batch_size:, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches
