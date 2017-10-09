import numpy as np


def initialize_parameters_zeros(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.

    Returns:
    parameters -- python dictionary containing all parameters Wl, bl.
    """
    np.random.seed(1)               # to get consistent output
    parameters = {}                 # initialize parameters dictionary
    L = len(layers_dims)            # number of layers in the network

    for l in range(1, L):
        parameters["W" + str(l)] = np.zeros(
            (layers_dims[l], layers_dims[l - 1]))
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


def initialize_parameters_random(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.

    Returns:
    parameters -- python dictionary containing all parameters Wl, bl.
    """
    np.random.seed(1)               # to get consistent output
    parameters = {}                 # initialize parameters dictionary
    L = len(layers_dims)            # number of layers in the network

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(
            layers_dims[l], layers_dims[l - 1]) * 10
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


def initialize_parameters_he_xavier(layers_dims, initialization_method="he"):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    initialization_method -- string specify the initialization method to be
                             used: "he", "xavier".

    Returns:
    parameters -- python dictionary containing all parameters Wl, bl
    """
    np.random.seed(1)               # to get consistent output
    parameters = {}                 # initialize parameters dictionary
    L = len(layers_dims)            # number of layers in the network

    if initialization_method == "he":
        for l in range(1, L):
            parameters["W" + str(l)] = np.random.randn(
                layers_dims[l],
                layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l - 1])
            parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
    elif initialization_method == "xavier":
        for l in range(1, L):
            parameters["W" + str(l)] = np.random.randn(
                layers_dims[l],
                layers_dims[l - 1]) * np.sqrt(1 / layers_dims[l - 1])
            parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


def model(X, Y, layers_dims, learning_rate=0.01, num_iterations=1000,
          print_cost=True, hidden_layers_activation_fn="relu",
          initialization_method="he"):
    """
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size.
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    hidden_layers_activation_fn -- activation function to be used on hidden
                                   layers, string: "tanh", "relu"
    initialization_method -- flag to choose which initialization to use
                             ("zeros","random", "he", or "xavier")

    Returns:
    parameters -- parameters learnt by the model.
    """
    # to get consistent results
    np.random.seed(1)

    # initialize cost list
    cost_list = []

    # initialize parameters
    if initialization_method == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)

    elif initialization_method == "random":
        parameters = initialize_parameters_random(layers_dims)

    else:
        parameters = initialize_parameters_he_xavier(
            layers_dims, initialization_method)

    # iterate over num_iterations
    for i in range(num_iterations):
        # iterate over L-layers to get the final output and the cache
        AL, caches = L_model_forward(
            X, parameters, hidden_layers_activation_fn)

        # compute cost to plot it
        cost = compute_cost(AL, Y)

        # iterate over L-layers backward to get gradients
        grads = L_model_backward(AL, Y, caches, hidden_layers_activation_fn)

        # update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # append each 100th cost to the cost list
        if (i + 1) % 100 == 0 and print_cost:
            print("The cost after {} iterations is: {}".format(i + 1, cost))

        if i % 100 == 0:
            cost_list.append(cost)

    # plot the cost curve
    plt.figure(figsize=(18, 12))
    plt.plot(cost_list)
    plt.ylabel("Cost")
    plt.title(
        "Cost curve: learning rate = {} and {} initialization method".format(
            learning_rate, initialization_method))

    return parameters


def accuracy(X, parameters, Y, activation_fn="relu"):
    """
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    parameters -- python dictionary containing all learnt parameters
    Y -- true "label" vector of shape (1, number of examples)
    activation_fn -- activation function to be used on hidden
                     layers, string: "tanh", "relu"

    Returns:
    accuracy -- accuracy rate after applying parameters on the input data
    """
    probs, caches = L_model_forward(X, parameters, activation_fn)
    labels = (probs > 0.5) * 1
    accuracy = np.mean(labels == Y) * 100

    return "The accuracy rate is: {:.2f}%.".format(accuracy)
