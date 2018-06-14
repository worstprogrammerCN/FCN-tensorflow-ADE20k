import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from Dataset import TestDataset
import FCN

# Define training setting.
NUM_OF_CLASSESS = 151
TEST_NUM = 100

# Define directories of checkpoint and data.
LOGS_DIR = "logs"
DATA_DIR = "Data_zoo/MIT_SceneParsing/ADEChallengeData2016"


def metrics(label: np.ndarray, pred: np.ndarray, num_class: int):
    """

    :param label: (h, w)
    :param pred:  (h, w)
    :param num_class:
    :return:
    """
    label = label.astype(np.uint8)
    pred = np.round(pred)
    mat: np.ndarray = np.zeros((num_class, num_class))
    [height, width] = label.shape
    for i in range(height):
        for j in range(width):
            pixel_label = label[i][j]
            pixel_pred = pred[i][j]
            mat[pixel_label][pixel_pred] += 1

    total_is = np.sum(mat, axis=1)  # 水平和
    total_n_jis = np.sum(mat, axis=0)  # 竖直和

    ious = []  # a list of intersection over union for each i
    t_n_ii = []  # the total number of pixels of class i that is correctly predicted
    total_iou_n_ii = []  # a list of (iou * n_ii) for each i
    total_n_ii_divied_t_i = []
    for i in range(1, num_class):  # calculate iou for class in [1, num_class]
        n_ii = mat[i][i]  # the number of pixels of class i that is correctly predicted

        total_i = total_is[i]  # the total number of pixels of class i
        total_n_ji = total_n_jis[i]  # the total number of pixels predicted to be class i

        # Ignore the category that doesn't show in the image.
        if total_i == 0:
            continue

        t_n_ii.append(n_ii)
        total_n_ii_divied_t_i.append(n_ii * 1.0 / total_i)

        # Calculate iou.
        if n_ii == 0:
            iou = 0  # intersection over union
        else:
            iou = (n_ii + 0.0) / (total_i + total_n_ji - n_ii)
        total_iou_n_ii.append(iou * total_i)
        ious.append(iou)
    # print(ious)

    pixel_acc = np.sum(t_n_ii) / np.sum(mat[1:, 1:])
    mean_acc = np.sum(total_n_ii_divied_t_i) / len(t_n_ii)
    mean_intersection_over_union = np.mean(np.array(ious))
    frequency_weighted_iu = np.sum(total_iou_n_ii) * 1.0 / np.sum(mat)

    return pixel_acc, mean_acc, mean_intersection_over_union, frequency_weighted_iu


def main(argv=None):
    # Define tensors.
    keep_probability = tf.placeholder(tf.float32, name="keep_probability")
    images = tf.placeholder(tf.float32, shape=[1, None, None, 3], name="input_image")
    pred_annotation, logits = FCN.get_fcn_8s_net(images, keep_probability, NUM_OF_CLASSESS)

    # Define test dataset.
    validation_dataset = TestDataset(DATA_DIR)

    # Define session config, init session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Initialize variables.
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Restore model.
    print("Setting up Saver...")
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(LOGS_DIR)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    # Define the four metrics.
    metric = {
        "p_acc": [],
        "m_acc": [],
        "miou": [],
        "wiou": []
    }

    # Test accuracy on the first 100 images.

    for i in range(TEST_NUM):
        image, annotation = validation_dataset.next_image()

        # Run the prediction.
        pred = sess.run(pred_annotation, feed_dict={images: image, keep_probability: 1.0})[0]

        # Squeeze data.
        pred = np.squeeze(pred, axis=2)
        annotation = np.squeeze(annotation[0], axis=2)

        # Evaluate metrics.
        pixel_acc, mean_acc, miou, weighted_iu = metrics(annotation, pred, NUM_OF_CLASSESS)
        metric["p_acc"].append(pixel_acc)
        metric["m_acc"].append(mean_acc)
        metric["miou"].append(miou)
        metric["wiou"].append(weighted_iu)

        # Print result.
        print("picture %d" % validation_dataset.cur_index, pixel_acc, mean_acc, miou, weighted_iu)

        # # Show good result.
        # # The prediction is over the average if pixel_acc > 0.66 and miou > 0.33
        # if pixel_acc > 0.66 and miou > 0.33:
        #     image = np.squeeze(image, axis=0)
        #     plt.subplot(3, 1, 1)
        #     plt.imshow(image)
        #
        #     plt.subplot(3, 1, 2)
        #     plt.imshow(pred)
        #
        #     plt.subplot(3, 1, 3)
        #     plt.imshow(annotation)
        #
        #     plt.show()

    # Calculate the mean of each metric.
    for key in metric:
        metric[key] = np.mean(metric[key])
        print("%s:" % key, metric[key])


if __name__ == '__main__':
    tf.app.run()
