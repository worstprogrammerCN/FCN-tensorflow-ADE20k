import numpy as np
import tensorflow as tf

import FCN_down_sizing


def deconv(inputs, filters, kernel_size, stride=2):
    return tf.layers.conv2d_transpose(inputs, filters, kernel_size, strides=[stride, stride], padding="same")


def max_pool_2x2(inputs):
    return tf.nn.max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def avg_pool_2x2(inputs):
    return tf.nn.avg_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def get_fcn_8s_net(image, keep_prob, num_of_class):
    """
    Construct FCN-8s net and return the prediction layers.

    :param image: numpy.ndarray, (batch_size, height, width, channel).
        The image or annotation to be processed.
    :param keep_prob: the keeping probability of dropout layer.
    :param num_of_class: Total num of classes, including the "other" class.
        When the dataset is ADE20k, its value is 151.

    :return:
        expanded_annotation_pred: Tensor, (batch_size, height, width, 1).
        The predicted class index of each pixel, by calculating tf.argmax on {conv_t3}.

        conv_t3: Tensor, (batch_size, height, width, num_of_class).
        The predicted probability of each class on pixels.
        The last output layer that hasn't calculate tf.argmax.

    """
    # The down sizing part of FCN.
    pool3, pool4, conv8 = FCN_down_sizing.getNet(image, keep_prob, num_of_class)

    # Fuse layer 1,
    pool4_shape = pool4.get_shape()
    conv_t1 = deconv(conv8, pool4_shape[3].value, [4, 4])
    fuse1 = tf.add(conv_t1, pool4)

    # Fuse layer 2.
    pool3_shape = pool3.get_shape()
    conv_t2 = deconv(fuse1, pool3_shape[3].value, [4, 4])
    fuse2 = tf.add(conv_t2, pool3)

    # Output layer.
    conv_t3 = deconv(fuse2, num_of_class, kernel_size=[16, 16], stride=8)  # image shape is  224 * 224

    # Predicted annotation without channel dimension.
    annotation_pred = tf.argmax(conv_t3, axis=3, name="prediction")

    # Predicted annotation with full dimension.
    expanded_annotation_pred = tf.expand_dims(annotation_pred, dim=3)

    return expanded_annotation_pred, conv_t3
