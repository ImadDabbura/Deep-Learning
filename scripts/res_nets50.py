import numpy as np
import keras as keras
from keras.layers import (Add, Activation,
                          AveragePooling2D,
                          BatchNormalization,
                          Conv2D, Dense,
                          Flatten, Input,
                          MaxPooling2D,
                          ZeroPadding2D)
from keras.models import Model
import tensorflow as tf


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """
    Implementation of the identity block.

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev).
    kernel_size -- tuple of 2 integers specifying the height and the width of
                   the 2D convolution window of the middle layer of the main
                   path.
    filters -- python list of integers, defining the number of filters in the
               CONV layers of the main path.
    stage -- integer, used to name the layers, depending on their position in
             the network.
    block -- string/character, used to name the layers, depending on their
             position in the network.

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    # Retrieve all filters
    f1, f2, f3 = filters

    # Define conv and bn names
    conv_name = "res" + str(stage) + block + "_branch"
    bn_name = "bn" + str(stage) + block + "_branch"

    # Build first layer
    X = Conv2D(filters=f1, kernel_size=(1, 1), strides=(1, 1),
               padding="valid", name=conv_name + "2a")(input_tensor)
    X = BatchNormalization(axis=3, name=bn_name + "2a")(X)
    X = Activation("relu")(X)

    # Build second layer
    X = Conv2D(filters=f2, kernel_size=kernel_size, strides=(1, 1),
               padding="same", name=conv_name + "2b")(X)
    X = BatchNormalization(axis=3, name=bn_name + "2b")(X)
    X = Activation("relu")(X)

    # Build third layer
    X = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1),
               padding="valid", name=conv_name + "2c")(X)
    X = BatchNormalization(axis=3, name=bn_name + "2c")(X)
    # Add input tensor to X
    X = Add()([X, input_tensor])
    # Apply activation fn
    X = Activation("relu")(X)

    return X


def conv_block(
        input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """
    Implementation of the convolutional block.

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev).
    kernel_size -- tuple of 2 integers specifying the height and the width of
                   the 2D convolution window of the middle layer of the main
                   path.
    filters -- python list of integers, defining the number of filters in the
    CONV layers of the main path.
    stage -- integer, used to name the layers, depending on their position in
    the network.
    block -- string/character, used to name the layers, depending on their
    position in the network.
    strides -- strides to be used in 1st conv and shortcut conv.

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C).
    """
    # Retrieve all filters
    f1, f2, f3 = filters

    # Define conv and bn names
    conv_name = "res" + str(stage) + block + "_branch"
    bn_name = "bn" + str(stage) + block + "_branch"

    # Build first layer
    X = Conv2D(filters=f1, kernel_size=(1, 1), strides=strides,
               padding="valid", name=conv_name + "2a")(input_tensor)
    X = BatchNormalization(axis=3, name=bn_name + "2a")(X)
    X = Activation("relu")(X)

    # Build second layer
    X = Conv2D(filters=f2, kernel_size=kernel_size, strides=(1, 1),
               padding="same", name=conv_name + "2b")(X)
    X = BatchNormalization(axis=3, name=bn_name + "2b")(X)
    X = Activation("relu")(X)

    # Build third layer
    X = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1),
               padding="valid", name=conv_name + "2c")(X)
    X = BatchNormalization(axis=3, name=bn_name + "2c")(X)
    # Apply conv and bn to shortcut
    X_shortcut = Conv2D(filters=f3, kernel_size=(1, 1), strides=strides,
                        padding="valid", name=conv_name + "1")(input_tensor)
    X_shortcut = BatchNormalization(axis=3, name=bn_name + "1")(X_shortcut)
    # Add input tensor to X
    X = Add()([X, X_shortcut])
    # Apply activation fn
    X = Activation("relu")(X)

    return X


def ResNet50(input_shape=None, classes=2):
    """
    Implementation of ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 ->
    CONVBLOCK -> IDBLOCK*3 -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK ->
    IDBLOCK*2 -> AVGPOOL -> TOPLAYER.

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras.
    """
    # Define input tensor
    input_tensor = Input(input_shape)

    # Zero padding
    X = ZeroPadding2D(padding=(3, 3))(input_tensor)

    # Build stage 1
    X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name="conv1")(X)
    X = BatchNormalization(axis=3, name="bn_conv1")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

    # Build stage 2
    X = conv_block(X, kernel_size=(3, 3), filters=[
                   64, 64, 256], stage=2, block="a", strides=(1, 1))
    X = identity_block(X, kernel_size=(3, 3), filters=[
                       64, 64, 256], stage=2, block="b")
    X = identity_block(X, kernel_size=(3, 3), filters=[
                       64, 64, 256], stage=2, block="c")

    # Build stage 3
    X = conv_block(X, kernel_size=(3, 3), filters=[
                   128, 128, 512], stage=3, block="a", strides=(2, 2))
    X = identity_block(X, kernel_size=(3, 3), filters=[
                       128, 128, 512], stage=3, block="b")
    X = identity_block(X, kernel_size=(3, 3), filters=[
                       128, 128, 512], stage=3, block="c")
    X = identity_block(X, kernel_size=(3, 3), filters=[
                       128, 128, 512], stage=3, block="d")

    # Build stage 4
    X = conv_block(X, kernel_size=(3, 3), filters=[
                   256, 256, 1024], stage=4, block="a", strides=(2, 2))
    X = identity_block(X, kernel_size=(3, 3), filters=[
                       256, 256, 1024], stage=4, block="b")
    X = identity_block(X, kernel_size=(3, 3), filters=[
                       256, 256, 1024], stage=4, block="c")
    X = identity_block(X, kernel_size=(3, 3), filters=[
                       256, 256, 1024], stage=4, block="d")
    X = identity_block(X, kernel_size=(3, 3), filters=[
                       256, 256, 1024], stage=4, block="e")
    X = identity_block(X, kernel_size=(3, 3), filters=[
                       256, 256, 1024], stage=4, block="f")

    # Build stage 5
    X = conv_block(X, kernel_size=(3, 3), filters=[
                   512, 512, 2048], stage=5, block="a", strides=(2, 2))
    X = identity_block(X, kernel_size=(3, 3), filters=[
                       512, 512, 2048], stage=5, block="b")
    X = identity_block(X, kernel_size=(3, 3), filters=[
                       512, 512, 2048], stage=5, block="c")

    # Average pooling
    X = AveragePooling2D(pool_size=(2, 2), name="avg_pool")(X)

    # Output layer
    X = Flatten()(X)
    X = Dense(units=classes, activation="softmax", name="fc" + str(classes))(X)

    # Instantiate the model
    model = Model(inputs=input_tensor, outputs=X, name="ResNets")

    return model
