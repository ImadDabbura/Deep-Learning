import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as keras


def nn_with_keras(
        X_train, Y_train, X_test, Y_test, num_classes=2, learning_rate=0.001,
        batch_size=64, verbose=0, vaildation_split=0, num_epochs=1000,
        activation_fun="tanh"):
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

    # Convert label vector to one_hot encoding
    y_one_hot = keras.utils.to_categorical(Y_train, num_classes)

    # Building keras model
    model = keras.models.Sequential()
    model.add(
        keras.layers.Dense(
            units=5,
            input_dim=num_features,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            activation=activation_fun
        )
    )

    model.add(
        keras.layers.Dense(
            units=5,
            input_dim=5,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            activation=activation_fun
        )
    )

    model.add(
        keras.layers.Dense(
            units=num_classes,
            input_dim=5,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            activation="softmax"
        )
    )

    sgd_optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(sgd_optimizer, loss="categorical_crossentropy")

    history = model.fit(X_train, y_one_hot, batch_size=batch_size,
                        epochs=num_epochs, verbose=verbose,
                        validation_split=validation_split)
    # Compute predictions
    train_preds = model.predict_classes(X_train, verbose=verbose)
    test_preds = model.predict_classes(X_test, verbose=verbose)

    # Compute accuracies
    train_accuracy = np.mean(Y_train == preds_train) * 100
    test_accuracy = np.mean(Y_test == preds_test) * 100

    # Print out accuracies
    print("The training accuracy rate is: {:.2f}%.".format(train_accuracy))
    print("The test accuracy rate is: {:.2f}%.".format(test_accuracy))
