from __future__ import print_function

import tensorflow as tf
import os

# Dataset Parameters - CHANGE HERE
IMAGE_PATH = './trainDataSet/class1/d.jpg' # the dataset file or root folder path.
MODEL_PATH='./model/model'
# Image Parameters
N_CLASSES = 2 # CHANGE HERE, total number of classes
IMG_HEIGHT = 64 # CHANGE HERE, the image height to be resized to
IMG_WIDTH = 64 # CHANGE HERE, the image width to be resized to
CHANNELS = 3 # The 3 color channels, change to 1 if grayscale


# -----------------------------------------------
# THIS IS A CLASSIC CNN (see examples, section 3)
# -----------------------------------------------
# Note that a few elements have changed (usage of queues).

# Parameters
learning_rate = 0.001
num_steps = 100
batch_size = 1
display_step = 10

# Network Parameters
dropout = 0.75 # Dropout, probability to keep units

# Build the data input
# Create model
def conv_net(x, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 32 filters and a kernel size of 5
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)
        # Because 'softmax_cross_entropy_with_logits' already apply softmax,
        # we only apply softmax to testing network
        out = tf.nn.softmax(out) if not is_training else out

    return out


# Because Dropout have different behavior at training and prediction time, we
# need to create 2 distinct computation graphs that share the same weights.


image_test = tf.read_file(IMAGE_PATH)
image_test = tf.image.decode_jpeg(image_test, channels=CHANNELS)

    # Resize images to a common size
image_test = tf.image.resize_images(image_test, [IMG_HEIGHT, IMG_WIDTH])

    # Normalize
image_test = image_test * 1.0/127.5 - 1.0
label=0

    # Create batches
X_test= tf.train.batch([image_test], batch_size=batch_size,
                          capacity=batch_size * 8,
                          num_threads=4)

# Create a graph for training
logits_train = conv_net(X_test, N_CLASSES, dropout, reuse=False, is_training=True)
# Create another graph for testing that reuse the same weights

# Evaluate model (with test logits, for dropout to be disabled)
result=tf.argmax(logits_train, 1)


# Saver object
saver = tf.train.Saver()

# Start training
with tf.Session() as sess:

    saver.restore(sess,MODEL_PATH)
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord)
    result1=sess.run(result)
    print(result1[0])