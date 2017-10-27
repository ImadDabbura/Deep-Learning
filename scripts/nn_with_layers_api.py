import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def nn_with_layers_api(
        X_train, Y_train, X_test, Y_test, num_classes=2, learning_rate=0.001,
        num_epochs=1000, activation_fun="tanh"):
    """
    Build 3 layer neural network using tensorflow high level API "layers".

    Arguments:
    X_train -- training input data, shape:
               number of examples x number of features.
    Y_train -- training label vector, shape: number of example.
    X_test -- test input data, shape: number of examples x number of features.
    Y_test -- test label vector, shape: number of example.
    num_classes -- number of classes for the classification problem:
                   --> scalar.
    learning_rate -- step size the gradient descent makes on each iteration:
                     --> scalar.
    num_epochs -- number of iterations learning algorithm goes over the data:
                  --> scalar.
    activation_fun -- non-linear function applied on hidden layers, string:
                      "tanh", "relu"

    Returns:
    Print both training and test accuracy.
    Plot the cost curve vs num_epochs.
    """
    num_features = X_train.shape[1]
    tf.set_random_seed(123)

    if activation_function == "tanh":
        activation_function = tf.tanh

    elif activation_function == "relu":
        activation_function = tf.nn.relu

    # Building the graph
    g = tf.Graph()
    with g.as_default():
        # Setup placeholders for X and Y
        X = tf.placeholder(dtype=tf.float32,
                           shape=(None, num_features),
                           name="X")
        Y = tf.placeholder(dtype=tf.int32,
                           shape=(None),
                           name="Y")

        # convert Y to one_hot
        y_one_hot = tf.one_hot(indices=Y, depth=num_classes)

        # Build hidden and output layers
        h1 = tf.layers.dense(inputs=X, units=5,
                             activation=activation_function,
                             name="layer1")
        h2 = tf.layers.dense(inputs=h1, units=5,
                             activation=activation_function,
                             name="layer2")
        logits = tf.layers.dense(inputs=h2, units=2,
                                 activation=None,
                                 name="layer3")

        # Define loss function and the optimizer
        loss = tf.losses.softmax_cross_entropy(onehot_labels=y_one_hot,
                                               logits=logits)
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate).minimize(loss)

        # Define probabilities and predictions
        probabilities = tf.nn.softmax(logits=logits, name="probabilities")
        predictions = tf.argmax(probabilities, axis=1)

        # Initialize the variables
        init = tf.global_variables_initializer()

    # Run the computational graph
    with tf.Session(graph=g) as sess:
        costs = []
        sess.run(init)

        for i in range(3000):
            _, cost = sess.run([optimizer, loss],
                               feed_dict={X: X_train, Y: Y_train})

            if i % 100 == 0:
                costs.append(cost)

        # Compute the probabilities and predictions of both train & test data
        preds_train = sess.run(predictions, feed_dict={X: X_train})
        preds_test = sess.run(predictions, feed_dict={X: X_test})

        # Print out accuracy
        train_accuracy = np.mean(Y_train == preds_train) * 100
        test_accuracy = np.mean(Y_test == preds_test) * 100

        print("The training accuracy rate is: {:.2f}%.".format(train_accuracy))
        print("The test accuracy rate is: {:.2f}%.".format(test_accuracy))

        # Plot the cost curve
        plt.plot(costs)
        plt.xlabel("Epochs (per hundreds)")
        plt.ylabel("Loss")
        plt.title(
            r"Cost curve with learning rate $\alpha$= {}".format(
                learning_rate))
