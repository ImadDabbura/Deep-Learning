import os

import matplotlib.pyplot as plt
import numpy as np

# import local modules
from coding_deep_neural_network_from_scratch import (initialize_parameters,
                                                     L_model_forward,
                                                     compute_cost,
                                                     L_model_backward,
                                                     update_parameters,
                                                     accuracy)


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, shape: input size, number of examples
    Y -- "label" vector, shape: 1, number of examples
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    # shuffle training set
    permutation = np.random.permutation(m)
    shuffle_X = X[:, permutation]
    shuffle_Y = Y[:, permutation]

    num_complete_minibatches = m // mini_batch_size

    for k in range(num_complete_minibatches):
        mini_batch_X = shuffle_X[:, k*mini_batch_size:(k + 1)*mini_batch_size]
        mini_batch_Y = shuffle_Y[:, k*mini_batch_size:(k + 1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # check if there are some examples left if m % batch_size != 0
    if m % mini_batch_size != 0:
        mini_batch_X = shuffle_X[:, num_complete_minibatches*mini_batch_size:]
        mini_batch_Y = shuffle_Y[:, num_complete_minibatches*mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def initialize_momentum(parameters):
    """
    Initializes the velocity as a python dictionary with:
                - keys: "dW1", "db1", ..., "dWL", "dbL"
                - values: numpy arrays of zeros of the same shape as the
                          corresponding gradients/parameters.

    Arguments:
    parameters -- python dictionary containing parameters.

    Returns:
    v -- python dictionary containing the current velocity.
    """
    L = len(parameters) // 2
    v = {}

    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        v["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])

    return v


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    Update parameters using Momentum

    Arguments:
    parameters -- python dictionary containing your parameters:
    grads -- python dictionary containing your gradients for each parameters:
    v -- python dictionary containing the current velocity:
    beta -- the momentum hyperparameter --> scalar
    learning_rate -- the learning rate --> scalar

    Returns:
    parameters -- python dictionary containing your updated parameters
    v -- python dictionary containing your updated velocities
    """
    L = len(parameters) // 2

    for l in range(1, L + 1):
        # update momentum velocity
        v["dW" + str(l)] =\
            beta * v["dW" + str(l)] + (1 - beta) * grads["dW" + str(l)]
        v["db" + str(l)] =\
            beta * v["db" + str(l)] + (1 - beta) * grads["db" + str(l)]
        # update parameters
        parameters["W" + str(l)] =\
            parameters["W" + str(l)] - learning_rate * v["dW" + str(l)]
        parameters["b" + str(l)] =\
            parameters["b" + str(l)] - learning_rate * v["db" + str(l)]

    return parameters, v


def initialize_rmsprop(parameters):
    """
    Initializes the velocity as a python dictionary with:
                - keys: "dW1", "db1", ..., "dWL", "dbL"
                - values: numpy arrays of zeros of the same shape as the
                          corresponding gradients/parameters.

    Arguments:
    parameters -- python dictionary containing parameters.

    Returns:
    s -- python dictionary containing the current velocity.
    """
    L = len(parameters) // 2
    s = {}

    for l in range(1, L + 1):
        s["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        s["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])

    return s


def update_parameters_with_rmsprop(parameters, grads, s, beta, learning_rate,
                                   epsilon=1e-8):
    """
    Update parameters using Momentum

    Arguments:
    parameters -- python dictionary containing parameters:
    grads -- python dictionary containing gradients for each parameters:
    s -- python dictionary containing the current velocity:
    beta -- the momentum hyperparameter --> scalar
    learning_rate -- the learning rate --> scalar
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters
    v -- python dictionary containing updated velocities
    """
    L = len(parameters) // 2

    for l in range(1, L + 1):
        # update momentum velocity
        s["dW" + str(l)] =\
            beta * s["dW" + str(l)] +\
            (1 - beta) * np.square(grads["dW" + str(l)])
        s["db" + str(l)] =\
            beta * s["db" + str(l)] +\
            (1 - beta) * np.square(grads["db" + str(l)])
        # update parameters
        parameters["W" + str(l)] -= (learning_rate * grads["dW" + str(l)])\
            / (np.sqrt(s["dW" + str(l)] + epsilon))
        parameters["b" + str(l)] -= (learning_rate * grads["db" + str(l)])\
            / (np.sqrt(s["db" + str(l)] + epsilon))

    return parameters, s


def initialize_adam(parameters):
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL"
                - values: numpy arrays of zeros of the same shape as the
                          corresponding gradients/parameters.

    Arguments:
    parameters -- python dictionary containing your parameters.

    v -- python dictionary that will contain the exponentially weighted
         average of the gradient.
    s -- python dictionary that will contain the exponentially weighted
         average of the squared gradient.
    """
    L = len(parameters) // 2
    v = {}
    s = {}

    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        v["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
        s["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        s["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])

    return v, s


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using Adam

    Arguments:
    parameters -- python dictionary containing parameters:
    grads -- python dictionary containing gradients for each parameters:
    v -- Adam variable, moving average of the first gradient
    s -- Adam variable, moving average of the squared gradient
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates
    beta2 -- Exponential decay hyperparameter for the second moment estimates
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing updated parameters
    v -- Adam variable, moving average of the first gradient
    s -- Adam variable, moving average of the squared gradient
    """
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}

    for l in range(1, L + 1):
        # update the moving avergae of both first gradient and squared gradient
        v["dW" + str(l)] = beta1 * v["dW" + str(l)] +\
            (1 - beta1) * grads["dW" + str(l)]
        v["db" + str(l)] = beta1 * v["db" + str(l)] +\
            (1 - beta1) * grads["db" + str(l)]
        s["dW" + str(l)] = beta2 * s["dW" + str(l)] +\
            (1 - beta2) * np.square(grads["dW" + str(l)])
        s["db" + str(l)] = beta2 * s["db" + str(l)] + \
            (1 - beta2) * np.square(grads["db" + str(l)])

        # compute the corrected-bias estimate of the moving averages
        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - beta1**t)
        v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - beta1**t)
        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - beta2**t)
        s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - beta2**t)

        # update parameters
        parameters["W" + str(l)] -= (
            learning_rate * v_corrected["dW" + str(l)])\
            / (np.sqrt(s_corrected["dW" + str(l)] + epsilon))
        parameters["b" + str(l)] -= (
            learning_rate * v_corrected["db" + str(l)])\
            / (np.sqrt(s_corrected["db" + str(l)] + epsilon))

    return parameters, v, s


def model(X, Y, layers_dims, optimizer="adam", learning_rate=0.01,
          mini_batch_size=64, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8,
          num_epochs=3000, print_cost=True, activation_fn="relu"):
    """
    Implements multi-neural network model which can be run in different
    optimizer modes.

    Arguments:
    X -- input data, shape: number of features, number of examples
    Y -- label vector, shape: 1, number of examples
    layers_dims -- python list, containing the size of each layer
    optimizer -- "mb", "momentum", "rmsprop", or "adam".
    learning_rate -- the learning rate --> scalar.
    mini_batch_size -- the size of a mini batch
    beta -- Momentum/RMSProp hyperparameter
    beta1 -- Exponential decay hyperparameter for the past gradients
    beta2 -- Exponential decay hyperparameter for the past squared gradients
    epsilon -- hyperparameter preventing division by zero
    num_epochs -- number of epochs
    print_cost -- True to print the cost every 1000 epochs
    activation_fn -- function to be used on hidden layers: "relu", or "tanh"

    Returns:
    parameters -- python dictionary containing updated parameters
    """
    # set random seed to get consistent output
    seed = 1
    np.random.seed(seed)

    # initialize parameters
    parameters = initialize_parameters(layers_dims)

    # initialize moving averages based on optimizer modes
    assert(optimizer == "mb" or optimizer == "momentum" or
           optimizer == "rmsprop" or optimizer == "adam")

    if optimizer == "momentum":
        v = initialize_momentum(parameters)

    elif optimizer == "rmsprop":
        s = initialize_rmsprop(parameters)

    elif optimizer == "adam":
        v, s = initialize_adam(parameters)
        t = 0

    # initialize costs list
    costs = []

    # iterate over number of epochs
    for epoch in range(num_epochs):
        # split the training data into mini batches
        seed += 1
        mini_batches = random_mini_batches(X, Y, mini_batch_size, seed=seed)

        # iterate over mini batches
        for mini_batch in mini_batches:
            mini_batch_X, mini_batch_Y = mini_batch

            # compute fwd prop
            AL, caches = L_model_forward(
                mini_batch_X, parameters, activation_fn)

            # compute cost
            cost = compute_cost(AL, mini_batch_Y)

            # compute gradients
            grads = L_model_backward(AL, mini_batch_Y, caches, activation_fn)

            # update parameters
            if optimizer == "mb":
                parameters = update_parameters(
                    parameters, grads, learning_rate)

            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(
                    parameters, grads, v, beta, learning_rate)

            elif optimizer == "rmsprop":
                parameters, s = update_parameters_with_rmsprop(
                    parameters, grads, s, beta, learning_rate, epsilon)

            elif optimizer == "adam":
                t += 1
                parameters, v, s = update_parameters_with_adam(
                    parameters, grads, v, s, t, learning_rate, beta1, beta2,
                    epsilon)

        # compute epoch cost
        AL, caches = L_model_forward(
            X_train, parameters, activation_fn)
        cost = compute_cost(AL, Y_train)

        if epoch % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(costs)
    plt.ylabel('Cost')
    plt.xlabel('Epochs (per hundreds)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters
