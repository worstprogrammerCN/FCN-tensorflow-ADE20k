import tensorflow as tf

# The FCN down sizing part contains 2 part.

# 1. Define the former part of downsizing, using
# the first 15 convolutional layers of vgg-19.


def conv3x3(inputs, filters, name):
    return tf.layers.conv2d(inputs=inputs,
                            filters=filters,
                            kernel_size=[3, 3],
                            padding="same",
                            activation=tf.nn.relu,
                            name=name)


def avg_pool_2x2(inputs):
    return tf.nn.avg_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def _vgg(image):
    """
        Derived from vgg-19 net, but it's not complete. This net only use the first 15 conv layers of vgg-19.
    """

    conv1_1 = conv3x3(image, 64, "conv1_1")
    conv1_2 = conv3x3(conv1_1, 64, "conv1_2")
    pool1 = avg_pool_2x2(conv1_2)

    conv2_1 = conv3x3(pool1, 128, "conv2_1")
    conv2_2 = conv3x3(conv2_1, 128, "conv2_2")
    pool2 = avg_pool_2x2(conv2_2)

    conv3_1 = conv3x3(pool2, 256, "conv3_1")
    conv3_2 = conv3x3(conv3_1, 256, "conv3_2")
    conv3_3 = conv3x3(conv3_2, 256, "conv3_3")
    conv3_4 = conv3x3(conv3_3, 256, "conv3_4")
    pool3 = avg_pool_2x2(conv3_4)

    conv4_1 = conv3x3(pool3, 512, "conv4_1")
    conv4_2 = conv3x3(conv4_1, 512, "conv4_2")
    conv4_3 = conv3x3(conv4_2, 512, "conv4_3")
    conv4_4 = conv3x3(conv4_3, 512, "conv4_4")
    pool4 = avg_pool_2x2(conv4_4)

    conv5_1 = conv3x3(pool4, 512, "conv5_1")
    conv5_2 = conv3x3(conv5_1, 512, "conv5_2")
    conv5_3 = conv3x3(conv5_2, 512, "conv5_3")

    return pool3, pool4, conv5_3


# 2. Define the last part of FCN downsizing.
def conv7x7(inputs, filters, name):
    return tf.layers.conv2d(inputs=inputs,
                            filters=filters,
                            kernel_size=[7, 7],
                            padding="same",
                            activation=tf.nn.relu,
                            name=name)


def conv1x1(inputs, filters, name):
    return tf.layers.conv2d(inputs=inputs,
                            filters=filters,
                            kernel_size=[1, 1],
                            padding="same",
                            activation=tf.nn.relu,
                            name=name)


def getNet(image, keep_prob, num_of_class):
    """
    Construct the downsizing part of FCN and return it.
    :param image: numpy.ndarray, (batch_size, height, width, channel).
        The image or annotation to be processed. 
    :param keep_prob: the keeping probability of dropout layer.
    :param num_of_class: Total num of classes, including the "other" class.
        When the dataset is ADE20k, its value is 151.
    :return: 
    """
    pool3, pool4, conv5_3 = _vgg(image)
    pool5 = avg_pool_2x2(conv5_3)

    conv6 = conv7x7(pool5, 4096, "conv6")
    dropout6 = tf.nn.dropout(conv6, keep_prob=keep_prob)

    conv7 = conv1x1(dropout6, 4096, "conv7")
    dropout7 = tf.nn.dropout(conv7, keep_prob=keep_prob)

    conv8 = conv1x1(dropout7, num_of_class, "conv8")

    return pool3, pool4, conv8
