import os

import tensorflow as tf
import numpy as np
from skimage import io

from Dataset import ImageReader
import FCN

# Define training settings.
NUM_OF_CLASSESS = 151
BATCH_SIZE = 2
LEARNING_RATE = 1e-4

# Define directories of checkpoint, data and model.
LOGS_DIR = "logs"
DATA_DIR = "Data_zoo/MIT_SceneParsing/ADEChallengeData2016"
MODEL_DIR = "MODEL_ZOO"

# Define image and output directory.
IMAGE_DIR = "infer"
OUTPUT_DIR = "output"


def metrics(label: np.ndarray, pred: np.ndarray, num_class: np.uint8):
    label = label.astype(np.uint8)
    pred = np.round(pred)
    mat: np.ndarray = np.zeros((num_class, num_class))
    [height, width] = label.shape
    for i in range(height):
        for j in range(width):
            pixel_label = label[i][j]
            pixel_pred = pred[i][j]
            mat[pixel_label][pixel_pred] += 1

    total_n_jis = np.sum(mat, axis=0)
    total_is = np.sum(mat, axis=1)

    ious = []  # a list of intersection over union for each i
    t_n_ii = []  # the total number of pixels of class i that is correctly predicted
    total_iou_n_ii = []  # a list of (iou * n_ii) for each i
    total_n_ii_divied_t_i = []
    for i in range(1, num_class):  # calculate iou for class in [1, num_class]
        n_ii = mat[i][i]  # the number of pixels of class i that is correctly predicted

        total_i = total_is[i]  # the total number of pixels of class i
        total_n_ji = total_n_jis[i]  # the total number of pixels predicted to be class i
        if total_i == 0:
            continue

        t_n_ii.append(n_ii)
        total_n_ii_divied_t_i.append(n_ii * 1.0 / total_i)

        if n_ii == 0:
            iou = 0 # intersection over union
        else:
            iou = (n_ii + 0.0) / (total_i + total_n_ji - n_ii)
        total_iou_n_ii.append(iou * total_i)
        ious.append(iou)
    # print(ious)

    pixel_acc = np.sum(t_n_ii) / np.sum(mat)
    mean_acc = np.sum(total_n_ii_divied_t_i) / len(t_n_ii)
    mean_intersection_over_union = np.mean(np.array(ious))
    frequency_weighted_iu = np.sum(total_iou_n_ii) * 1.0 / np.sum(mat)

    return pixel_acc, mean_acc, mean_intersection_over_union, frequency_weighted_iu


def main(argv=None):
    # Define tensors.
    keep_probability = tf.placeholder(tf.float32, name="keep_probability")
    images = tf.placeholder(tf.float32, shape=[1, None, None, 3], name="input_image")
    pred_annotation, logits = FCN.get_fcn_8s_net(images, keep_probability, NUM_OF_CLASSESS)

    # Define image reader.
    print("setting up ImageReader...")
    image_reader = ImageReader(IMAGE_DIR)

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

    while image_reader.has_next:
        # Session Calculate predicted annotation.
        image = image_reader.next_image()
        index = image_reader.cur_index
        fd = {images: image, keep_probability: 1.0}
        preds: np.ndarray = sess.run(pred_annotation, feed_dict=fd)
        print(preds.shape)
        pred = np.squeeze(preds[0], axis=2)

        # Save prediction as file.
        output_path = os.path.join(OUTPUT_DIR, "prediction_" + str(index) + ".png")
        io.imsave(output_path, pred.astype(np.uint8))
        print("Saved image: %s" % output_path)


if __name__ == '__main__':
    tf.app.run()
