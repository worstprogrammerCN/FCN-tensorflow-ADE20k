import datetime
import os

import tensorflow as tf

from Dataset import TrainDataset, TestDataset
import FCN

# Define training setting.
MAX_STEPS = int(1e5 + 1)
NUM_OF_CLASSESS = 151
BATCH_SIZE = 2
LEARNING_RATE = 1e-5

# Define directories of checkpoint, data and model.
LOGS_DIR = "logs"
DATA_DIR = "Data_zoo/MIT_SceneParsing/ADEChallengeData2016"
MODEL_DIR = "MODEL_ZOO"


def train(loss_val, var_list):
    """
    Define the optimization layer.
    :return: Tensor. The tensor of training operation.
    """
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    return optimizer.apply_gradients(grads)


def main(argv=None):
    # Define loss tensor operations.
    keep_probability = tf.placeholder(tf.float32, name="keep_probability")
    images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_image")
    annotations = tf.placeholder(tf.int32, shape=[None, None, None, 1], name="annotations")

    pred_annotation, logits = FCN.get_fcn_8s_net(images, keep_probability, NUM_OF_CLASSESS)
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotations,
                                                                                            squeeze_dims=[3]),
                                                                          name="entropy")))

    # Define train tensor operations.
    trainable_var = tf.trainable_variables()
    train_op = train(loss, trainable_var)
    tf.summary.scalar('loss', loss)

    # Define the summary writer
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(LOGS_DIR, 'train'))

    # Initialize the two dataset.
    print("setting up dataset...")
    dataset = TrainDataset(DATA_DIR)
    validation_dataset = TestDataset(DATA_DIR)

    # Define tensorflow session config.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Initialize variables.
    sess.run(tf.global_variables_initializer())

    # Restore model.
    print("Setting up Saver...")
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(LOGS_DIR)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    # Train for MAX_STEPS steps.
    for step in range(MAX_STEPS):
        # Train the loss.
        train_images, train_annotations = dataset.get_batch(BATCH_SIZE)
        feed_dict = {images: train_images, annotations: train_annotations, keep_probability: 0.8}

        # Calculate training loss and print.
        _, train_loss, summary = sess.run([train_op, loss, merged], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        print("Step: %d, Train_loss:%g, selected number:%d" % (step, train_loss, dataset.cur_index))

        if step % 500 == 0:
            # Calculate validation loss and print.
            valid_image, valid_annotations = validation_dataset.next_image()
            feed_dict = {images: valid_image, annotations: valid_annotations, keep_probability: 1.0}
            valid_loss = sess.run(loss, feed_dict=feed_dict)
            print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))

            # Save the model.
            print("--------------saving model %d----------------------" % step)
            saver.save(sess, os.path.join(LOGS_DIR, "model.ckpt"), step)
    train_writer.close()


if __name__ == '__main__':
    tf.app.run()
