# Loading packages
import os

import matplotlib.pyplot as plt
import numpy as np

# local modules
os.chdir("../scripts/")
from coding_deep_neural_network_from_scratch import (initialize_parameters,
                                                     linear_forward,
                                                     linear_activation_forward,
                                                     L_model_forward,
                                                     compute_cost,
                                                     relu_gradient,
                                                     sigmoid_gradient,
                                                     tanh_gradient,
                                                     update_parameters,
                                                     accuracy)
from gradient_checking import dictionary_to_vector


# Regularization (L2)
def compute_cost_reg(AL, Y, parameters, lambd=0):
    """
    Computes the Cross-Entropy cost function with L2 regularization.

    Arguments:
    AL -- post-activation, output of forward propagation, shape:
          output size x number of examples.
    Y --  labels vector, shape: output size x number of examples.
    parameters -- python dictionary containing parameters of the model.
    lambd -- regularization hyperparameter --> scalar.

    Returns:
    cost - value of the regularized loss function.
    """
    # number of examples
    m = Y.shape[1]

    # compute traditional cross entropy cost
    cross_entropy_cost = compute_cost(AL, Y)

    # convert parameters dictionary to vector
    parameters_vector = dictionary_to_vector(parameters)

    # compute the regularization penalty
    L2_regularization_penalty = (
        lambd / (2 * m)) * np.sum(np.square(parameters_vector))

    # compute the total cost
    cost = cross_entropy_cost + L2_regularization_penalty

    return cost


def linear_backword_reg(dZ, cache, lambd=0):
    """
    Computes the gradient of the output w.r.t weight, bias, and post-activation
    output of (l - 1) layers at layer l.
    
    Arguments:
    dZ -- Gradient of the cost with respect to the linear output
          (of current layer l).
    cache -- tuple of values (A_prev, W, b) coming from the forward
             propagation in the current layer.
    lambd -- regularization hyperparameter --> scalar.

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation
               (of the previous layer l-1).
    dW -- Gradient of the cost with respect to W (current layer l).
    db -- Gradient of the cost with respect to b (current layer l).
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T) + (lambd / m) * W
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward_reg(dA, cache, activation_fn="relu", lambd=0):
    """
    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache)
    activation -- the activation to be used in this layer, stored as a string:
                  "sigmoid", "tanh", or "relu"
    lambd -- regularization hyperparameter --> scalar.

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation
               (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l),
          same shape as W
    db -- Gradient of the cost with respect to b (current layer l),
          same shape as b
    """
    linear_cache, activation_cache = cache

    if activation_fn == "sigmoid":
        dZ = sigmoid_gradient(dA, activation_cache)
        dA_prev, dW, db = linear_backword_reg(dZ, linear_cache, lambd)

    elif activation_fn == "tanh":
        dZ = tanh_gradient(dA, activation_cache)
        dA_prev, dW, db = linear_backword_reg(dZ, linear_cache, lambd)

    elif activation_fn == "relu":
        dZ = relu_gradient(dA, activation_cache)
        dA_prev, dW, db = linear_backword_reg(dZ, linear_cache, lambd)

    return dA_prev, dW, db


def L_model_backward_reg(AL, Y, caches, hidden_layers_activation_fn="relu",
                         lambd=0):
    """
    Computes the gradient of output layer w.r.t weights, biases, etc. starting
    on the output layer in reverse topological order.
    
    Arguments:
    AL -- probability vector, output of the forward propagation
          (L_model_forward()).
    Y -- true vector.
    caches -- list of caches.
    hidden_layers_activation_fn -- activation function to be used on hidden
                                   layers, string: "tanh", "relu".
    lambd -- regularization hyperparameter --> scalar.

    Returns:
    grads -- A dictionary with the gradients.
    """
    Y = Y.reshape(AL.shape)
    L = len(caches)
    grads = {}

    dAL = np.divide(AL - Y, np.multiply(AL, 1 - AL))

    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] =\
        linear_activation_backward_reg(dAL, caches[L - 1], "sigmoid", lambd)

    for l in range(L - 1, 0, -1):
        current_cache = caches[l - 1]
        grads["dA" + str(l - 1)], grads["dW" + str(l)], grads["db" + str(l)] =\
            linear_activation_backward_reg(
                grads["dA" + str(l)], current_cache,
                hidden_layers_activation_fn, lambd)

    return grads


def model_with_regularization(
        X, Y, layers_dims, learning_rate=0.01, num_iterations=3000,
        print_cost=False, hidden_layers_activation_fn="relu", lambd=0,
        keep_prob=1):
    """
    Implements L-Layer neural network.

    Arguments:
    X -- input data, shape: num_px * num_px * 3 x number of examples.
    Y -- label vector, shape: 1 x number of examples.
    layers_dims -- list containing the input size and each layer size, of
                   length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule.
    num_iterations -- number of iterations of the optimization loop.
    print_cost -- if True, it prints the cost every 100 steps.
    hidden_layers_activation_fn -- activation function to be used on hidden
                                   layers, string: "tanh", "relu".
    lambd -- regularization hyperparameter --> scalar.
    keep_prob - probability of keeping a neuron active during drop-out, scalar.

    Returns:
    parameters -- parameters learnt by the model. They can then be used
                  to predict.
    """
    # get number of examples
    m = X.shape[1]

    # to get consistents output
    np.random.seed(1)

    # initialize parameters
    parameters = initialize_parameters(layers_dims)

    # intialize cost list
    cost_list = []

    # implement gradient descent
    for i in range(num_iterations):
        # compute forward propagation
        AL, caches = L_model_forward(
            X, parameters, hidden_layers_activation_fn)

        # compute regularized cost
        reg_cost = compute_cost_reg(AL, Y, parameters, lambd)

        # compute gradients
        grads = L_model_backward_reg(
            AL, Y, caches, hidden_layers_activation_fn, lambd)

        # update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # print cost
        if (i + 1) % 100 == 0 and print_cost:
            print("The regularized cost after {} iterations: {}".format(
                (i + 1), reg_cost))

        # append cost
        if i % 100 == 0:
            cost_list.append(reg_cost)

    # plot the cost curve
    plt.plot(cost_list)
    plt.xlabel("Iterations (per hundreds)")
    plt.ylabel("Cost")
    plt.title("Cost curve for the learning rate = {}".format(learning_rate))

    return parameters