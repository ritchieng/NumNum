import os
import sys
import tarfile
import pickle
import matplotlib.pyplot as plt
import numpy as np, h5py 
import pandas as pd
from PIL import Image
import tensorflow as tf
from sklearn.cross_validation import train_test_split


print('Loading pickled data...')

pickle_file = 'SVHN.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    X_train = save['train_dataset']
    y_train = save['train_labels']
    X_val = save['valid_dataset']
    y_val = save['valid_labels']
    X_test = save['test_dataset']
    y_test = save['test_labels']
    del save  
    print('Training data shape:', X_train.shape)
    print('Training label shape:',y_train.shape)
    print('Validation data shape:', X_val.shape)
    print('Validation label shape:', y_val.shape)
    print('Test data shape:', X_test.shape)
    print('Test label shape:', y_test.shape)

print('Data successfully loaded!')



print('Defining accuracy function...')
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 2).T == labels)
            / predictions.shape[1] / predictions.shape[0])
print('Accuracy function defined!')

# CNN Model
print('Loading data and building computation graph...')

'''Basic information'''
# We processed image size to be 32
image_size = 32
# Number of channels: 1 because greyscale
num_channels = 1
# Mini-batch size
batch_size = 16
# Number of output labels
num_labels = 11

'''Filters'''
# depth: number of filters (output channels) - should be increasing
# num_channels: number of input channels set at 1 previously
patch_size = 5
depth_1 = 16
depth_2 = depth_1 * 2
depth_3 = depth_2 * 3

# Number of hidden nodes in fully connected layer 1
num_hidden = 64
shape = [batch_size, image_size, image_size, num_channels]

graph = tf.Graph()

with graph.as_default():

    '''Input Data'''
    # X_train: (223965, 32, 32, 1)
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))

    # y_train: (223965, 7)
    tf_train_labels = tf.placeholder(
        tf.int32, shape=(batch_size, 6))

    # X_val: (11788, 32, 32, 1)
    tf_valid_dataset = tf.constant(X_val)

    # X_test: (13067, 32, 32, 1)
    tf_test_dataset = tf.constant(X_test)

    '''Variables'''

    # Create Variables Function
    def init_weights_conv(shape, name):
        return tf.get_variable(shape=shape, name=name,
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
    def init_weights_fc(shape, name):
        return tf.get_variable(shape=shape, name=name,
            initializer=tf.contrib.layers.xavier_initializer())

    def init_biases(shape, name):
        return tf.Variable(
            tf.constant(1.0, shape=shape),
            name=name
        )

    # Create Function for Image Size: Pooling
    # 3 Convolutions
    # 2 Max Pooling
    def output_size_pool(input_size, conv_filter_size, pool_filter_size,
                         padding, conv_stride, pool_stride):
        if padding == 'same':
            padding = -1.00
        elif padding == 'valid':
            padding = 0.00
        else:
            return None
        # After convolution 1
        output_1 = (
            ((input_size - conv_filter_size - 2 * padding) / conv_stride) + 1.00)
        # After pool 1
        output_2 = (
            ((output_1 - pool_filter_size - 2 * padding) / pool_stride) + 1.00)
        # After convolution 2
        output_3 = (
            ((output_2 - conv_filter_size - 2 * padding) / conv_stride) + 1.00)
        # After pool 2
        output_4 = (
            ((output_3 - pool_filter_size - 2 * padding) / pool_stride) + 1.00)
        # After convolution 2
        output_5 = (
            ((output_4 - conv_filter_size - 2 * padding) / conv_stride) + 1.00)
        # After pool 2
        # output_6 = (
        #     ((output_5 - pool_filter_size - 2 * padding) / pool_stride) + 1.00)
        return int(output_5)

    # Convolution 1
    # Input channels: num_channels = 1
    # Output channels: depth = depth_1
    w_c1 = init_weights_conv([patch_size, patch_size, num_channels, depth_1], 'w_c1')
    b_c1 = init_biases([depth_1], 'b_c1')

    # Convolution 2
    # Input channels: num_channels = depth_1
    # Output channels: depth = depth_2
    w_c2 = init_weights_conv([patch_size, patch_size, depth_1, depth_2], 'w_c2')
    b_c2 = init_biases([depth_2], 'b_c2')

    # Convolution 3
    # Input channels: num_channels = depth_2
    # Output channels: depth = depth_3
    w_c3 = init_weights_conv([patch_size, patch_size, depth_2, depth_3], 'w_c3')
    b_c3 = init_biases([depth_3], 'b_c3')

    # Fully Connect Layer 1
    final_image_size = output_size_pool(input_size=image_size,
                                        conv_filter_size=5, pool_filter_size=2,
                                        padding='valid', conv_stride=1,
                                        pool_stride=2)
    print('Final image size after convolutions {}'.format(final_image_size))
    w_fc1 = init_weights_fc([final_image_size*final_image_size*depth_3, num_hidden], 'w_fc1')
    b_fc1 = init_biases([num_hidden], 'b_fc1')

    # Softmax 1
    w_s1 = init_weights_fc([num_hidden, num_labels], 'w_s1')
    b_s1 = init_biases([num_labels], 'b_s1')

    # Softmax 2
    w_s2 = init_weights_fc([num_hidden, num_labels], 'w_s2')
    b_s2 = init_biases([num_labels], 'b_s2')

    # Softmax 3
    w_s3 = init_weights_fc([num_hidden, num_labels], 'w_s3')
    b_s3 = init_biases([num_labels], 'b_s3')

    # Softmax 4
    w_s4 = init_weights_fc([num_hidden, num_labels], 'w_s4')
    b_s4 = init_biases([num_labels], 'b_s4')

    # Softmax 5
    w_s5 = init_weights_fc([num_hidden, num_labels], 'w_s5')
    b_s5 = init_biases([num_labels], 'b_s5')

    def model(data, keep_prob, shape):
        with tf.name_scope("conv_layer_1"):
            conv_1 = tf.nn.conv2d(
                data, w_c1, strides=[1, 1, 1, 1], padding='VALID')
            hidden_conv_1 = tf.nn.relu(conv_1 + b_c1)
            pool_1 = tf.nn.max_pool(
                hidden_conv_1, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
        with tf.name_scope("conv_layer_2"):
            conv_2 = tf.nn.conv2d(
                pool_1, w_c2, strides=[1, 1, 1, 1], padding='VALID')
            hidden_conv_2 = tf.nn.relu(conv_2 + b_c2)
            pool_2 = tf.nn.max_pool(
                hidden_conv_2, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
        with tf.name_scope("conv_layer_3"):
            conv_3 = tf.nn.conv2d(
                pool_2, w_c3, strides=[1, 1, 1, 1], padding='VALID')
            hidden_conv_3 = tf.nn.relu(conv_3 + b_c3)
        with tf.name_scope("fc_layer_1"):
            hidden_drop = tf.nn.dropout(hidden_conv_3, keep_prob)
            shape = hidden_drop.get_shape().as_list()
            reshape = tf.reshape(
                hidden_drop, [shape[0], shape[1] * shape[2] * shape[3]])
            hidden_fc = tf.nn.relu(
                tf.matmul(reshape, w_fc1) + b_fc1)
        with tf.name_scope("softmax_1"):
            logits_1 = tf.matmul(hidden_fc, w_s1) + b_s1
        with tf.name_scope("softmax_2"):
            logits_2 = tf.matmul(hidden_fc, w_s2) + b_s2
        with tf.name_scope("softmax_3"):
            logits_3 = tf.matmul(hidden_fc, w_s3) + b_s3
        with tf.name_scope("softmax_4"):
            logits_4 = tf.matmul(hidden_fc, w_s4) + b_s4
        with tf.name_scope("softmax_5"):
            logits_5 = tf.matmul(hidden_fc, w_s5) + b_s5
        return [logits_1, logits_2, logits_3, logits_4, logits_5]

    '''Training Computation'''
    [logits_1, logits_2, logits_3, logits_4, logits_5] = model(
        tf_train_dataset, 0.9, shape)

    '''Loss Function'''
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                   logits_1, tf_train_labels[:, 1])) + \
               tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                   logits_2, tf_train_labels[:, 2])) + \
               tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                   logits_3, tf_train_labels[:, 3])) + \
               tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                   logits_4, tf_train_labels[:, 4])) + \
               tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                   logits_5, tf_train_labels[:, 5]))
        # Add scalar summary for cost
        tf.scalar_summary("loss", loss)

    '''Optimizer'''
    # Decaying learning rate
    # count the number of steps taken
    global_step = tf.Variable(0)
    start_learning_rate = 0.05
    learning_rate = tf.train.exponential_decay(
        start_learning_rate, global_step, 10000, 0.96)

    optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(
        loss, global_step=global_step)

    '''Predictions'''
    def softmax_combine(dataset, shape):
        train_prediction = tf.pack([
            tf.nn.softmax(model(dataset, 1.0, shape)[0]),
            tf.nn.softmax(model(dataset, 1.0, shape)[1]),
            tf.nn.softmax(model(dataset, 1.0, shape)[2]),
            tf.nn.softmax(model(dataset, 1.0, shape)[3]),
            tf.nn.softmax(model(dataset, 1.0, shape)[4])])
        return train_prediction

    train_prediction = softmax_combine(tf_train_dataset, shape)
    valid_prediction = softmax_combine(tf_valid_dataset, shape)
    test_prediction = softmax_combine(tf_test_dataset, shape)

    '''Save Model (will be initiated later)'''
    saver = tf.train.Saver()

    '''Histogram for Weights'''
    # Add histogram summaries for weights
    tf.histogram_summary("w_c1_summ", w_c1)
    tf.histogram_summary("b_c1_summ", b_c1)

    tf.histogram_summary("w_c2_summ", w_c2)
    tf.histogram_summary("b_c2_summ", b_c2)

    tf.histogram_summary("w_c3_summ", w_c3)
    tf.histogram_summary("b_c3_summ", b_c3)

    tf.histogram_summary("w_fc1_summ", w_fc1)
    tf.histogram_summary("b_fc1_summ", b_fc1)

    tf.histogram_summary("w_s1_summ", w_s1)
    tf.histogram_summary("b_s1_summ", b_s1)

    tf.histogram_summary("w_s2_summ", w_s2)
    tf.histogram_summary("b_s2_summ", b_s2)

    tf.histogram_summary("w_s3_summ", w_s3)
    tf.histogram_summary("b_s3_summ", b_s3)

    tf.histogram_summary("w_s4_summ", w_s4)
    tf.histogram_summary("b_s4_summ", b_s4)

    tf.histogram_summary("w_s5_summ", w_s5)
    tf.histogram_summary("b_s5_summ", b_s5)

print('Data loaded and computation graph built!')

num_steps = 60000

print('Running computation and iteration...')
print('If you are unable to save the summary, please change the path to where you want it to write.')

with tf.Session(graph=graph) as session:
    writer = tf.train.SummaryWriter("log_trial_2", session.graph)  # for 0.8
    merged = tf.merge_all_summaries()

    '''If you want to restore model'''
    # saver.restore(session, "model_trial_1.ckpt")
    # print("Model restored!")

    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (y_train.shape[0] - batch_size)
        batch_data = X_train[offset:(offset + batch_size), :, :, :]
        batch_labels = y_train[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data,
                     tf_train_labels: batch_labels}
        _, l, predictions, summary = session.run([optimizer, loss, train_prediction, merged],
                                        feed_dict=feed_dict)
        writer.add_summary(summary)
        if (step % 500 == 0):
            print(('Minibatch loss at step {}: {}').format(step, l))
            print(
            ('Minibatch accuracy: {}%'.format(accuracy(predictions, batch_labels[:,1:6]))))
            print(
            ('Validation accuracy: {}%'.format(accuracy(valid_prediction.eval(),
                                                     y_val[:,1:6]))))
    print(
    ('Test accuracy: {}%'.format(accuracy(test_prediction.eval(), y_test[:,1:6]))))

    save_path = saver.save(session, "model_trial_2.ckpt")
    print('Model saved in file: {}'.format(save_path))


print('Successfully completed computation and iterations!')

print('To view Tensorboard\'s visualizations, please run \
\'tensorboard --logdir=log_trial_2\' in your terminal')


