from __future__ import division, print_function, absolute_import

# Import Training data
import getTrainingData as gtrd
import getTestingData as gted
import tensorflow as tf
import numpy as np

# Training Parameters
learning_rate = 0.001
num_steps = 2000
batch_size = 128

# Network Parameters
num_classes = 2 #Total classes (0-1)
dropout = 0.5 # Dropout, probability to keep units


# Create the neural network
def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        x = x_dict['images']

        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 32, 32, 1])

        # Convolution Layer with 96 filters and a kernel size of 9
        conv1 = tf.layers.conv2d(x, 96, 9, padding='same',  activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 3
        conv1 = tf.layers.max_pooling2d(conv1, 3, 2)

        # Convolution Layer with 256 filters and a kernel size of 5
        conv2 = tf.layers.conv2d(conv1, 256, 5, padding='same', activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 3
        conv2 = tf.layers.max_pooling2d(conv2, 3, 2)

        # Convolution Layer with 384 filters and a kernel size of 3
        conv3 = tf.layers.conv2d(conv2, 384, 3, padding='same',  activation=tf.nn.relu)

        # Convolution Layer with 384 filters and a kernel size of 3
        conv4 = tf.layers.conv2d(conv3, 384, 3, padding='same',  activation=tf.nn.relu)

        # Convolution Layer with 256 filters and a kernel size of 3
        conv5 = tf.layers.conv2d(conv4, 256, 3, padding='same',  activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 3
        conv5 = tf.layers.max_pooling2d(conv5, 3, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv5)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 4096)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Fully connected layer (in tf contrib folder for now)
        fc2 = tf.layers.dense(fc1, 4096)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc2 = tf.layers.dropout(fc2, rate=dropout, training=is_training)
         
        # Output layer, class prediction
        out = tf.layers.dense(fc2, n_classes)

    return out


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):

    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = conv_net(features, num_classes, dropout, reuse=False,
                            is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True,
                           is_training=False)

    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)), name="loss_op")
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes, name="acc_op")
    
    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs

tf.logging.set_verbosity(tf.logging.INFO)

# Build the Estimator
model = tf.estimator.Estimator(model_fn)

# print(type(gtd.getImages()))
# print(gtd.getImages())
# print(type(gtd.getLabels()))
# print(gtd.getLabels())
imgs = gtrd.getImages()
lbls = gtrd.getLabels()

imgs2 = gted.getImages()
lbls2 = gted.getLabels()

# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': imgs}, y=lbls,
    batch_size=batch_size, num_epochs=None, shuffle=True)

# Train the Model
# model.train(input_fn, steps=num_steps, hooks=[logging_hook])
model.train(input_fn, steps=num_steps)

# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': imgs2}, y=lbls2,
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
e = model.evaluate(input_fn)
print("Testing Accuracy:", e['accuracy'])