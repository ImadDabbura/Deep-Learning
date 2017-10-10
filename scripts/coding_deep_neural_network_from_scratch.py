import numpy as np
import matplotlib.pyplot as plt


# Initialize parameters
def initialize_parameters(layers_dims):
    """
    Arguments:
    layers_dims -- list or numpy array that has the dimensions of each layer
                   in the network

    Returns:
    parameters -- dictionary that has the weight matrices and the bias vector
                  for each layer
    """
    np.random.seed(1)               # to get consistent output
    parameters = {}                 # initialize parameters dictionary
    L = len(layers_dims)             # number of layers in the network

    for l in range(1, L):           # we dont count input layer
        parameters["W" + str(l)] = np.random.randn(
            layers_dims[l], layers_dims[l - 1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

        assert(parameters["W" + str(l)].shape == (
            layers_dims[l], layers_dims[l - 1]))
        assert(parameters["b" + str(l)].shape == (layers_dims[l], 1))

    return parameters


# define activation functions that will be used in forward propagation
def sigmoid(Z):
    """
    Arguments:
    Z -- Output of linear layer

    Returns:
    A -- output of sigmoid
    cache -- return Z; it'll be useful for backpropagation
    """
    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, Z


def tanh(Z):
    """
    Arguments:
    Z -- Output of linear layer

    Returns:
    A -- output of sigmoid
    cache -- return Z; it'll be useful for backpropagation
    """
    A = np.tanh(Z)
    cache = Z

    return A, Z


def relu(Z):
    """
    Arguments:
    Z -- Output of linear layer

    Returns:
    A -- output of ReLU
    cache -- return Z; it'll be useful for backpropagation
    """
    A = np.maximum(0, Z)
    cache = Z

    return A, Z


def leaky_relu(Z):
    """
    Arguments:
    Z -- Output of linear layer

    Returns:
    A -- output of ReLU
    cache -- return Z; it'll be useful for backpropagation
    """
    A = np.maximum(0.1 * Z, Z)
    cache = Z

    return A, Z


# define helper functions that will be used in L-model forward prop
def linear_forward(A_prev, W, b):
    """
    Arguments:
    A_prev -- activations output from previous layer
    W -- Weight matrix, shape: size of current layer x size of previuos layer
    b -- bias vector, shape: size of current layer x 1

    Returns:
    Z -- Input of activation function
    cache -- tuple that stores A_prev, W, b to be used in backpropagation
    """
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation_fn):
    """
    Arguments:
    A_prev -- activations output from previous layer
    W -- Weight matrix, shape: size of current layer x size of previuos layer
    b -- bias vector, shape: size of current layer x 1
    activation_fn -- string that specify the activation function to be used:
                     "sigmoid", "tanh", "relu"

    Returns:
    A -- Output of the activation function
    cache -- tuple that stores linear_cache and activation_cache
             ((A_prev, W, b), Z) to be used in backpropagation
    """
    if activation_fn == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation_fn == "tanh":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = tanh(Z)

    elif activation_fn == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert(A.shape == (W.shape[0], A_prev.shape[1]))

    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters, hidden_layers_activation_fn="relu"):
    """
    Arguments:
    X -- Input matrix of shape input_size x training_examples
    parameters -- dictionary that contains all the weight matrices and bias
                  vectors for all layers
    hidden_layers_activation_fn -- activation function to be used on hidden
                                   layers, string: "tanh", "relu"

    Returns:
    AL -- probability vector of shape 1 x training_examples
    caches -- list that contains L tuples where each layer has: A_prev, W, b, Z
    """
    A = X                      # since input matrix A0
    caches = []                     # initialize the caches list
    L = len(parameters) // 2        # number of layer in the network

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(
            A_prev, parameters["W" + str(l)], parameters["b" + str(l)],
            activation_fn=hidden_layers_activation_fn)
        caches.append(cache)

    AL, cache = linear_activation_forward(
        A, parameters["W" + str(L)], parameters["b" + str(L)],
        activation_fn="sigmoid")
    caches.append(cache)

    assert(AL.shape == (1, X.shape[1]))

    return AL, caches


# compute cross-entropy cost
def compute_cost(AL, Y):
    """
    Arguments:
    AL -- probability vector of shape 1 x training_examples
    Y -- true "label" vector

    Returns:
    cost -- cross-entropy cost
    """
    m = Y.shape[1]              # number of examples
    cost = - (1 / m) * np.sum(
        np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))

    return cost


# define derivative of activation functions w.r.t z that will be used in
# back-propagation
def sigmoid_gradient(dA, Z):
    """
    Arguments:
    dA -- post-activation gradient, of any shape
    Z -- Input used for the activation fn on this layer

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    A, Z = sigmoid(Z)
    dZ = dA * A * (1 - A)

    return dZ


def tanh_gradient(dA, Z):
    """
    Arguments:
    dA -- post-activation gradient, of any shape
    Z -- Input used for the activation fn on this layer

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    A, Z = tanh(Z)
    dZ = dA * (1 - np.square(A))

    return dZ


def relu_gradient(dA, Z):
    """
    Arguments:
    dA -- post-activation gradient, of any shape
    Z -- Input used for the activation fn on this layer

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    dZ = np.array(dA, copy=True)
    dZ[dZ <= 0] = 0

    return dZ


# define helper functions that will be used in L-model back-prop
def linear_backword(dZ, cache):
    """
    Arguments:
    dZ -- Gradient of the cost with respect to the linear output
          (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward
             propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation
               (of the previous layer l-1)
    dW -- Gradient of the cost with respect to W (current layer l)
    db -- Gradient of the cost with respect to b (current layer l)
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation_fn):
    """
    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache)
    activation -- the activation to be used in this layer, stored as a string:
                  "sigmoid", "tanh", or "relu"

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
        dA_prev, dW, db = linear_backword(dZ, linear_cache)

    elif activation_fn == "tanh":
        dZ = tanh_gradient(dA, activation_cache)
        dA_prev, dW, db = linear_backword(dZ, linear_cache)

    elif activation_fn == "relu":
        dZ = relu_gradient(dA, activation_cache)
        dA_prev, dW, db = linear_backword(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches, hidden_layers_activation_fn="relu"):
    """
    Arguments:
    AL -- probability vector, output of the forward propagation
          (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches
    hidden_layers_activation_fn -- activation function to be used on hidden
                                   layers, string: "tanh", "relu"

    Returns:
    grads -- A dictionary with the gradients
    """
    Y = Y.reshape(AL.shape)
    L = len(caches)
    grads = {}

    dAL = np.divide(AL - Y, np.multiply(AL, 1 - AL))

    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads[
        "db" + str(L)] = linear_activation_backward(
            dAL, caches[L - 1], "sigmoid")

    for l in range(L - 1, 0, -1):
        current_cache = caches[l - 1]
        grads["dA" + str(l - 1)], grads["dW" + str(l)], grads[
            "db" + str(l)] = linear_activation_backward(
                grads["dA" + str(l)], current_cache,
                hidden_layers_activation_fn)

    return grads


# define the function to update both weight matrices and bias vectors
def update_parameters(parameters, grads, learning_rate):
    """
    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing parameters
    """
    L = len(parameters) // 2

    for l in range(1, L + 1):
        parameters["W" + str(l)] = parameters[
            "W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters[
            "b" + str(l)] - learning_rate * grads["db" + str(l)]

    return parameters


# define the multi-layer model using all the helper functions we wrote before
def L_layer_model(
        X, Y, layers_dims, learning_rate=0.01, num_iterations=3000,
        print_cost=False, hidden_layers_activation_fn="relu"):
    """
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape
         (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of
                   length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    hidden_layers_activation_fn -- activation function to be used on hidden
                                   layers, string: "tanh", "relu"

    Returns:
    parameters -- parameters learnt by the model. They can then be used
                  to predict.
    """
    # to get consistents output
    np.random.seed(1)

    # initialize parameters
    parameters = initialize_parameters(layers_dims)

    # intialize cost list
    cost_list = []

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
    plt.title("Cost curve for the learning rate = {}".format(learning_rate))

    return parameters
