import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_x -- number of input's features --> scalar.
    n_y -- number of classes --> scalar.

    Returns:
    X -- placeholder for the data input, shape: (n_x, None) & dtype "float".
    Y -- placeholder for the input labels, shape: (n_y, None) & dtype "int".
    """
    X = tf.placeholder(dtype=tf.float32,
                       shape=(n_x, None),
                       name="X")
    Y = tf.placeholder(dtype=tf.float32,
                       shape=(n_y, None),
                       name="Y")

    return X, Y


def initialize_parameters(layers_dims):
    """
    Initializes parameters to build a neural network with tensorflow.

    Arguments:
    layers_dims -- list of the dimension of each layers.

    Returns:
    parameters -- a dictionary of tensors containing weight matrices and bias.
    """
    L = len(layers_dims)
    parameters = {}

    for l in range(1, L):
        parameters["W" + str(l)] = tf.get_variable(
            name=("W" + str(l)), shape=(layers_dims[l], layers_dims[l - 1]),
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
        parameters["b" + str(l)] = tf.get_variable(
            name=("b" + str(l)), shape=(layers_dims[l], 1),
            dtype=tf.float32, initializer=tf.zeros_initializer())

        assert parameters["W" + str(l)].get_shape() == (layers_dims[l],
                                                        layers_dims[l - 1])
        assert parameters["b" + str(l)].get_shape() == (layers_dims[l], 1)

    return parameters


def convert_one_hot(Y, num_classes):
    """
    Creates one hot matrix.

    Arguments:
    Y -- label vector, shape: 1 x number of examples.
    num_classes -- number of classes --> scalar.

    Returns:
    y_one_hot -- one matrix.
    """
    # Convert to 1D array
    Y = Y.ravel()

    with tf.Session() as sess:
        y_one_hot = tf.one_hot(
            indices=Y, depth=num_classes, axis=0, name="y_one_hot")
        y_one_hot = sess.run(y_one_hot)

    return y_one_hot


def forward_propagation(X, parameters, activation_fn):
    """
    Implements the forward propagation for the model:

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples).
    parameters -- python dictionary containing your parameters.
    activation_fn -- activation function used on hidden layers, string:
                     "tanh" or "relu".

    Returns:
    ZL -- the output of the last LINEAR unit.
    """
    L = len(parameters) // 2
    m = X.get_shape().as_list()[1]
    output = {}
    output["A" + str(0)] = X

    if activation_fn == "tanh":
        activation_function = tf.tanh

    elif activation_fn == "relu":
        activation_function = tf.nn.relu

    for l in range(1, L):
        A_prev = output["A" + str(l - 1)]
        output["Z" + str(l)] = tf.add(
            tf.matmul(parameters["W" + str(l)], A_prev),
            parameters["b" + str(l)], name=("Z" + str(l)))
        output["A" + str(l)] = activation_function(output["Z" + str(l)])

    output["Z" + str(L)] = tf.add(
        tf.matmul(parameters["W" + str(L)], output["A" + str(L - 1)]),
        parameters["b" + str(L)], name=("Z" + str(L)))

    return output["Z" + str(L)]


def compute_cost(ZL, Y):
    """
    Computes the cross-entropy average cost.

    Arguments:
    ZL -- output of forward propagation (output of the last LINEAR unit).
    Y -- labels vector placeholder, same shape as Z3.

    Returns:
    cost - Tensor of the cost function.
    """
    logits = tf.transpose(ZL)
    labels = tf.transpose(Y)

    cost = tf.losses.softmax_cross_entropy(labels, logits)

    return cost


def model(X_train, Y_train, X_test, Y_test, layers_dims, learning_rate=0.0001,
          num_epochs=1500, minibatch_size=32, print_cost=True,
          activation_fn="relu"):
    """
    Implements a three-layer tensorflow neural network:

    Arguments:
    X_train -- training input data, shape:
               num of features x num of training examples.
    Y_train -- training label vector, shape:
               output size, number of training examples.
    X_test -- test input data, shape: num of features x num of test examples.
    Y_test -- test label vector, shape: output size, number of test examples.
    layers-dims -- list of the size of each layer.
    learning_rate -- step size of the gradient.
    num_epochs -- number of epochs of the optimization loop.
    minibatch_size -- size of a minibatch.
    print_cost -- True to print the cost every 100 epochs.
    activation_fn -- activation function used on hidden layers, string:
                     "tanh" or "relu".

    Returns:
    parameters -- parameters learnt by the model.
    """
    n_x = X_train.shape[0]
    n_y = Y_train.shape[0]
    costs = []
    seed = 1
    tf.set_random_seed(seed)

    # Instantiate a graph object
    g = tf.Graph()

    # Built the computational graph
    with g.as_default():
        # Create placeholders
        X, Y = create_placeholders(n_x, n_y)

        # Initialize parameters
        parameters = initialize_parameters(layers_dims)

        # Computes forward prop
        ZL = forward_propagation(X, parameters, activation_fn)

        # Computes cost
        loss = compute_cost(ZL, Y)

        # Define optimizer
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate).minimize(loss)

        # Initialize global variables
        init = tf.global_variables_initializer()

        # Save the the graph
        file_writer = tf.summary.FileWriter(logdir="./logs/", graph=g)

    # Run the computational graph
    with tf.Session(graph=g) as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            mini_batches = random_mini_batches(
                X_train, Y_train, minibatch_size, seed)
            num_minibatches = len(mini_batches)
            seed += 1
            epoch_cost = 0

            for mini_batch in mini_batches:
                mini_batch_X, mini_batch_Y = mini_batch
                _, mini_batch_cost = sess.run(
                    [optimizer, loss],
                    feed_dict={X: mini_batch_X, Y: mini_batch_Y})

                epoch_cost += mini_batch_cost / num_minibatches

            if epoch % 100 == 0 and print_cost:
                print("The cost after {} epochs is: {}".format(
                    epoch, epoch_cost))

            if epoch % 100 == 0:
                costs.append(epoch_cost)

        # Save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Compute the correct predictions
        correct_prediction = tf.equal(tf.argmax(ZL), tf.argmax(Y))

        # Compute accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # Print out the accuracy rates of training and test sets
        print("The training accuracy rate is: {:.2f}%".format(
            accuracy.eval(feed_dict={X: X_train, Y: Y_train}) * 100))
        print("The test accuracy rate is: {:.2f}%".format(
            accuracy.eval(feed_dict={X: X_test, Y: Y_test}) * 100))

    # plot the cost
    plt.plot(costs)
    plt.ylabel('Cost')
    plt.xlabel('Epochs (per hundreds)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters
