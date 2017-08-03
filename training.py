# By @Kevin Xu
# kevin28520@gmail.com
# Youtube: https://www.youtube.com/channel/UCVCSn4qQXTDAtGWpWAe4Plw
#
# The aim of this project is to use TensorFlow to process our own data.
#    - input_data.py:  read in data and generate batches
#    - model: build the model architecture
#    - training: train

# I used Ubuntu with Python 3.5, TensorFlow 1.0*, other OS should also be good.
# With current settings, 10000 traing steps needed 50 minutes on my laptop.

# data: cats vs. dogs from Kaggle
# Download link: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data
# data size: ~540M

# How to run?
# 1. run the training.py once
# 2. call the run_training() in the console to train the model.

# Note: 
# it is suggested to restart your kenel to train the model multiple times 
# (in order to clear all the variables in the memory)
# Otherwise errors may occur: conv1/weights/biases already exist......


# %%
import numpy as np
import tensorflow as tf
import model
from PIL import Image


#
def get_one_image(imgDir):
    #    '''Randomly pick one image from training data
    #    Return: ndarray
    #    '''
    image = Image.open(imgDir)
    image = image.resize([208, 208])
    image = np.array(image)
    return image  # %%


def evaluate_one_image(ckpt, imgDir):
    #    '''Test one image against the saved models and parameters
    #    '''
    #
    #    # you need to change the directories to yours.
    image_array = get_one_image(imgDir)

    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 2

        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 208, 208, 3])
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)

        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[208, 208, 3])

        # you need to change the directories to yours.


        saver = tf.train.Saver()

        with tf.Session() as sess:
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                # print('Loading success, global_step is %s' % global_step)
                # else:
                # print('No checkpoint file found')

            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
        return max_index, prediction
        # if max_index == 0:
        #    print('This is a cat with possibility %.6f' % prediction[:, 0])
        # else:
        #    print('This is a dog with possibility %.6f' % prediction[:, 1])  # %%
