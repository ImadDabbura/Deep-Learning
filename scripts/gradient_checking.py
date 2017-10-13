# Importing modules/packages
import os

import numpy as np

# Importing helper functions from local module
from coding_deep_neural_network_from_scratch import (L_model_forward,
                                                     compute_cost,
                                                     L_model_backward)


def dictionary_to_vector(dictionary):
    """
    Roll all dictionary into a single vector.
    """
    count = 0

    for key in dictionary.keys():
        new_vector = np.reshape(dictionary[key], (-1, 1))

        if count == 0:
            theta_vector = new_vector

        else:
            theta_vector = np.concatenate((theta_vector, new_vector), axis=0)

        count += 1

    return theta_vector


def vector_to_dictionary(vector, layers_dims):
    """
    Unroll parameters vector to dictionary using layers dimensions.

    Arguments:
    vector -- parameters vector
    layers_dims -- list or numpy array that has the dimensions of each layer
                   in the network.

    Returns:
    parameters -- python dictionary containing all parameters
    """
    L = len(layers_dims)
    parameters = {}
    k = 0

    for l in range(1, L):
        # create temp variable to store dimension used on each layer
        w_dim = layers_dims[l] * layers_dims[l - 1]
        b_dim = layers_dims[l]

        # create temporary var to be used in slicing parameters vector
        temp_dim = k + w_dim

        # add parameters to the dictionary
        parameters["W" + str(l)] = vector[k:temp_dim].reshape(
            layers_dims[l], layers_dims[l - 1])
        parameters["b" + str(l)] = vector[
            temp_dim:temp_dim + b_dim].reshape(b_dim, 1)

        k += w_dim + b_dim

    return parameters


def gradients_to_vector(gradients):
    """
    Roll all gradients into a single vector containing only dW and db
    """
    # get the number of indices for the gradients to iterate over
    valid_grads = [key for key in gradients.keys()
                   if not key.startswith("dA")]
    L = len(valid_grads) // 2
    count = 0

    # iterate over all gradients and append them to new_grads list
    for l in range(1, L + 1):

        if count == 0:
            new_grads = gradients["dW" + str(l)].reshape(-1, 1)
            new_grads = np.concatenate(
                (new_grads, gradients["db" + str(l)].reshape(-1, 1)), axis=0)

        else:
            new_grads = np.concatenate(
                (new_grads, gradients["dW" + str(l)].reshape(-1, 1)), axis=0)
            new_grads = np.concatenate(
                (new_grads, gradients["db" + str(l)].reshape(-1, 1)), axis=0)

        count += 1

    return new_grads


def forward_prop_cost(X, parameters, Y, hidden_layers_activation_fn="tanh"):
    """
    Implements the forward propagation and computes the cost.

    Arguments:
    X -- input data of shape number of features x number of examples.
    parameters -- python dictionary containing all parameters.
    Y -- true "label" of shape 1 x number of examples.
    hidden_layers_activation_fn -- activation function to be used on hidden
                                   layers,string: "tanh", "relu"

    Returns:
    cost -- cross-entropy cost.
    """
    # compute forward prop
    AL, caches = L_model_forward(X, parameters, hidden_layers_activation_fn)

    # compute cost
    cost = compute_cost(AL, Y)

    return cost


def gradient_check_n(
        parameters, gradients, X, Y, layers_dims, epsilon=1e-7,
        hidden_layers_activation_fn="tanh"):
    """
    Checks if back_prop computes correctly the gradient of the cost output by
    forward_prop.

    Arguments:
    parameters -- python dictionary containing all parameters.
    gradients -- output of back_prop, contains gradients of the cost ww.r.t
    the parameters.
    X -- input data of shape number of features x number of examples.
    Y -- true "label" of shape 1 x number of examples.
    epsilon -- tiny shift to the input to compute approximate gradient
    layers_dims -- list or numpy array that has the dimensions of each layer
                   in the network.

    Returns:
    difference -- difference between approx gradient and back_prop gradient
    """

    # roll out parameters and gradients dictionaries
    parameters_vector = dictionary_to_vector(parameters)
    gradients_vector = gradients_to_vector(gradients)

    # create vector of zeros to be used with epsilon
    grads_approx = np.zeros_like(parameters_vector)

    for i in range(len(parameters_vector)):
        # compute cost of theta + epsilon
        theta_plus = np.copy(parameters_vector)
        theta_plus[i] = theta_plus[i] + epsilon
        j_plus = forward_prop_cost(
            X, vector_to_dictionary(theta_plus, layers_dims), Y,
            hidden_layers_activation_fn)

        # compute cost of theta - epsilon
        theta_minus = np.copy(parameters_vector)
        theta_minus[i] = theta_minus[i] - epsilon
        j_minus = forward_prop_cost(
            X, vector_to_dictionary(theta_minus, layers_dims), Y,
            hidden_layers_activation_fn)

        # compute numerical gradients
        grads_approx[i] = (j_plus - j_minus) / (2 * epsilon)

    # compute the difference of numerical and analytical gradients
    numerator = np.linalg.norm(gradients_vector - grads_approx)
    denominator = np.linalg.norm(grads_approx) +\
    np.linalg.norm(gradients_vector)
    difference = numerator / denominator

    if difference > 10e-7:
        print("\033[31m" + "There is a mistake in back-propagation",
              "implementation. The difference is: {}".format(difference))

    else:
        print("\033[32m" + "There implementation of back-propagation is fine!",
              "The difference is: {}".format(difference))

    return difference
