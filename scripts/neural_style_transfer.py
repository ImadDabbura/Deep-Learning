import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from scipy.misc import imsave


# Download the VGG-19 model from here:
# http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat


def generate_noise_image(content_image, noise_ratio=0.2):
    """
    Generates a noisy image by adding random noise to the content_image.

    Arguments:
    content_image -- content image matrix.
    noise_ratio -- noise added to the content image and randomly generated
                   pixels from uniform distribution.

    Returns:
    noise_image -- randomly generated image.
    """
    # Retrieve shape of content image
    n_H, n_W, n_C = content_image.shape

    # Create random pixels from uniform distribution
    noise_img = np.random.uniform(-20, 20, size=((1,) + (n_H, n_W, n_C)))

    # Add noise to random pixels and content image
    noise_img = noise_img * noise_ratio + (1 - noise_ratio) * content_image

    return noise_img


def reshape_and_normalize_image(image, means):
    """
    Reshape image to match VGG model input: (1, n_H, n_W, n_C).
    """
    img = np.reshape(image, newshape=((1,) + image.shape))
    img = img - means

    return img


def save_image(image, path):
    """
    Save the generated image.
    """
    # Get rid of the first dimension
    img = image[0]

    # Make sure the values between 0 and 255
    img = np.clip(img, 0, 255).astype("uint8")

    imsave(path, img)


def compute_content_cost(a_C, a_G):
    """
    Computes the content cost.

    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations
           representing content of the image C.
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations
        representing content of the image G.

    Returns:
    J_content -- scalar that you compute using equation 1 above.
    """
    # Retrieve shape
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape content and generated tensors
    a_C = tf.transpose(tf.reshape(a_C, [n_H * n_W, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, [n_H * n_W, n_C]))

    # Compute the cost
    cost = (1 / (4 * n_H * n_W * n_C)) * \
        tf.reduce_sum(tf.squared_difference(a_C, a_G))

    return cost


def gram_matrix(A):
    """
    Computes the "Gram Matrix" of an image.

    Argument:
    A -- matrix of shape (n_C, n_H*n_W).

    Returns:
    G -- Gram matrix of A, of shape (n_C, n_C).
    """
    G = tf.matmul(A, tf.transpose(A))

    return G


def compute_layer_style_cost(a_S, a_G):
    """
    Computes the style cost at layer l.

    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations
           representing style of the image S.
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations
           representing style of the image G.

    Returns:
    J_style_layer -- style cost at layer l --> scalar value tensor.
    """
    # Retrieve shape
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape content and generated tensors
    a_S = tf.transpose(tf.reshape(a_S, [n_H * n_W, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, [n_H * n_W, n_C]))

    # Compute gram matrices for both style and generated tensors
    gram_S = gram_matrix(a_S)
    gram_G = gram_matrix(a_G)

    # Compute cost at layer l
    cost = (1 / (2 * n_H * n_W * n_C) ** 2) * \
        tf.reduce_sum(tf.squared_difference(gram_S, gram_G))

    return cost


def compute_style_cost(sess, model, style_layers):
    """
    Computes the overall style cost from several chosen layers.

    Arguments:
    model -- tensorflow model.
    style_layers -- list containing:
                    - names of the layers we would like to extract style from.
                    - a coefficient for each of them.

    Returns:
    J_style -- total style cost --> scalar value tensor.
    """
    J_style = 0

    # Loop over style layers
    for layer_name, weight in style_layers:
        # Select layers response
        output = model[layer_name]

        # Run the response tensor
        a_S = sess.run(output)

        # Assign the response tensor to generated image without evaluating it
        # because it will be evaluated last when running the graph
        a_G = output

        # Compute the cost at the layer
        cost = compute_layer_style_cost(a_S, a_G)

        # Increment the J_style cost
        J_style += weight * cost

    return J_style


def total_cost(J_content, J_style, alpha=10, beta=40):
    """
    Computes the total cost function.

    Arguments:
    J_content -- content cost.
    J_style -- style cost.
    alpha -- hyperparameter weighting the importance of the content cost.
    beta -- hyperparameter weighting the importance of the style cost.

    Returns:
    J -- total cost.
    """
    total_cost = alpha * J_content + beta * J_style

    return total_cost


def load_vgg_model(path, image_dim):
    """
    Load VGG-19 model.

    Arguments:
    path -- directory where the model is located including name of the file.
    image_dim -- dimension used to create input tenson of the model, of shape:
                 1 x n_H x n_W x n_C.
    Returns:
    graph -- model using all VGG-19 model except fully connected layers.
    """
    vgg_model = loadmat(path)
    vgg_layers = vgg_model["layers"]

    def weights(layer, expected_layer_name):
        """
        Returns the weights and bias for the layer given.
        """
        # Retrieve weights and bias
        w = vgg_layers[0][layer][0][0][2][0][0]
        b = vgg_layers[0][layer][0][0][2][0][1]

        # Reshape bias
        b = np.reshape(b, b.shape[0])

        # Retrieve layer name
        layer_name = vgg_layers[0][layer][0][0][0][0]

        assert layer_name == expected_layer_name

        return w, b

    def conv2d_relu(prev_layer_output, layer, layer_name):
        """
        Returns the Conv + ReLU using weights and bias from VGG-19 model at
        layer "layer".
        """
        # Retrieve weights and bias
        w, b = weights(layer, layer_name)

        # Apply Conv ops
        conv = tf.nn.conv2d(prev_layer_output, filter=w,
                            strides=(1, 1, 1, 1), padding="SAME") + b

        # Apply ReLU
        output = tf.nn.relu(conv)

        return output

    def avg_pool(prev_layer_output):
        """
        Returns Average Pooling on the previous layer output.
        """
        return tf.nn.avg_pool(prev_layer_output, ksize=(1, 2, 2, 1),
                              strides=(1, 2, 2, 1), padding="SAME")

    # Constructs the graph model
    graph = {}

    # generated image pixels will be updated on each iteration (training step)
    # Therefore, we use the input as tf variable
    graph['input'] = tf.Variable(
        np.zeros(shape=(image_dim), dtype='float32')
    graph['conv1_1'] = conv2d_relu(graph['input'], 0, 'conv1_1')
    graph['conv1_2'] = conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = avg_pool(graph['conv1_2'])
    graph['conv2_1'] = conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2'] = conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = avg_pool(graph['conv2_2'])
    graph['conv3_1'] = conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2'] = conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3'] = conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4'] = conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = avg_pool(graph['conv3_4'])
    graph['conv4_1'] = conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2'] = conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3'] = conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4'] = conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = avg_pool(graph['conv4_4'])
    graph['conv5_1'] = conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2'] = conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3'] = conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4'] = conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = avg_pool(graph['conv5_4'])

    return graph


def neural_style_transfer_model(
        input_image, content_image, style_image, content_layer,
        style_layers, alpha=10, beta=40, iterations=100, learning_rate=2,
        path="output/generated_image.jpg"):
    """
    Apply Neural Style Transfer on content and style images.

    Arguments:
    content_layer -- layers selected to use from content image.
    style_layers -- layers selected from style image.
    alpha -- weight of the content in generated image.
    beta -- weight of the style image in generated image.
    iterations -- number if iterations the gradient descent will train.
    learning_rate -- step size the gradient on each iteration.
    path -- path where the final generated image will be stored.

    Returns:
    Save the generated image to the specified directory using path.
    """
    # Means used when VGG-19 was trained
    means = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))

    # Reshape content and stye images
    content_image = reshape_and_normalize_image(content_image, means)
    style_image = reshape_and_normalize_image(style_image, means)

    # Run interactive tensorflow session
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    # Load VGG-19 model
    model = load_vgg_model(
        "pre-trained_model/imagenet-vgg-verydeep-19.mat")

    # Assign content image as input to the model
    sess.run(model["input"].assign(content_image))

    # Select the output tensor to the layer we selected
    output = model[content_layer]

    # Run the content layer
    a_C = sess.run(output)

    # Assign the content layer tensor to a_G
    a_G = output

    # Compute content cost
    J_content = compute_content_cost(a_C, a_G)

    # Assign style image as input to the model
    sess.run(model["input"].assign(style_image))

    # Compute style cost
    J_style = compute_style_cost(sess, model, style_layers)

    # Compute total cost
    J = total_cost(J_content, J_style, alpha, beta)

    # Define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(J)

    # Initialize all global variables in the graph
    sess.run(tf.global_variables_initializer())

    # Assign input image as input to the graph
    generated_image = sess.run(model["input"].assign(input_image))

    # loop over number of iterations
    for i in range(iterations):
        sess.run(optimizer)
        generated_image = sess.run(model["input"])

        # Print every 20 iteration
        if i % 20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration {}:".format(i))
            print("Total cost = {}".format(Jt))
            print("Content cost = {}".format(Jc))
            print("Style cost = {}".format(Js))

    # Save the generated image
    save_image(generated_image, path)
