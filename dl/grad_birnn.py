import tensorflow as tf
from tensorflow.python.ops.constant_op import constant
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np

import os
import sys
import time
from util.preprocess_data import FEATURE_IDXS_DICT, FEATURE_NAMES_DICT


# FIXME: Parameters
TAG = "A"
INPUT_DIM = 64
LABEL_NAME = "having_a_meal"

IMAGE_HEIGHT = 1
IMAGE_WIDTH = INPUT_DIM
NUM_INPUT_CHANNELS = 1
NUM_CLASSES = 2

LEARNING_RATE = 0.001
NUM_EPOCHS = 10
BATCH_SIZE = 128
NUM_HIDDENS = 128
DISPLAY_STEP = 100
VALIDATION_FREQUENCY = 1000

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "data")
TRAIN_DATA_PATH = os.path.join(DATA_DIR,
                               "integrated_data_%s_I%d_%s_train.dat" %
                               (TAG, INPUT_DIM, LABEL_NAME))
TEST_DATA_PATH = os.path.join(DATA_DIR,
                               "integrated_data_%s_I%d_%s_test.dat" %
                               (TAG, INPUT_DIM, LABEL_NAME))

TRAIN_SET_RATIO = 7
VALID_SET_RATIO = 1

VALIDATION_DATA_RATIO = float(VALID_SET_RATIO) / \
                        (TRAIN_SET_RATIO + VALID_SET_RATIO)

FEATURE_IDXS = []
for s in TAG:   # investigate TAG, character by character
  FEATURE_IDXS.append(FEATURE_IDXS_DICT[s])
FEATURE_IDXS = list(sum(FEATURE_IDXS, ()))  # flatten the list
NUM_FEATURES = 0
for i in range(len(FEATURE_IDXS)):
  if i % 2 == 0:
    NUM_FEATURES -= FEATURE_IDXS[i]
  else:
    NUM_FEATURES += FEATURE_IDXS[i]

# Network parameters
n_input = IMAGE_HEIGHT	  # Sensor data input (shape: 1*64)
n_steps = INPUT_DIM*NUM_FEATURES  # timesteps
n_hidden = NUM_HIDDENS  # number of features in hidden layers
n_classes = NUM_CLASSES   # Behavior total classes (not doing, doing)


def extract_data(filename, num_features, num_classes,
                 image_height, image_width,
                 input_num_feature_maps,
                 num_instances=None):
    """
    Extract the instances and labels.
    :param filename:
    :param num_instances:
    :return:
        data: A 4D tensor [num_instances, image_height,
                           num_features*image_width, input_num_feature_maps]
        labels: A 1-hot matrix [num_instances, NUM_CLASSES]
    """
    print("Extracting", filename)
    data_set = np.genfromtxt(filename, delimiter=',', dtype=np.float32)

    data = data_set[:, 2:-1]    # Exclude profile_id, timestamp, label
    profile_ids = data_set[:, 0]
    labels = data_set[:, -1]

    # Slice out the features and labels with the size of num_instances
    if not num_instances:
        num_instances = data.shape[0]
    data = data[:num_instances]
    profile_ids = profile_ids[:num_instances]
    labels = labels[:num_instances]
    print data.shape
    print num_instances, image_height, num_features*image_width, 1
    data = data.reshape((num_instances, image_height,
                         num_features*image_width, 1))
    data = data.repeat(input_num_feature_maps, axis=3)

    # Convert to dense 1-hot representation.
    labels = (np.arange(num_classes) == labels[:, None]).astype(np.float32)

    return data, profile_ids, labels


# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
# TensorFlow LSTM cell requires 2x n_hidden length (state & cell)
istate_fw = tf.placeholder("float", [None, 2*n_hidden])
istate_bw = tf.placeholder("float", [None, 2*n_hidden])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
  # Hidden layer weights => 2*n_hidden because of forward + backward cells
  "hidden": tf.Variable(tf.random_normal([n_input, 2*n_hidden])),
  "out": tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
}
biases = {
  "hidden": tf.Variable(tf.random_normal([2*n_hidden])),
  "out": tf.Variable(tf.random_normal([n_classes]))
}

def BiRNN(_X, _istate_fw, _istate_bw, _weights, _biases, _batch_size, _seq_len):

  # BiRNN requires to supply sequence_length as [batch_size, int64]
  # Note: Tensorflow 0.6.0 requires BiRNN sequence_length parameter to be set
  # For a better implementation with latest version of tensorflow, check below
  _seq_len = tf.fill([_batch_size], constant(_seq_len, dtype=tf.int64))

  # input shape: (batch_size, n_steps, n_input)
  _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
  # Reshape to prepare input to hidden activation
  _X = tf.reshape(_X, [-1, n_input]) # (n_steps*batch_size, n_input)
  # Linear activation
  _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']

  # Define lstm cells with tensorflow
  # Forward direction cell
  lstm_fw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
  # Backward direction cell
  lstm_bw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
  # Split data because rnn cell needs a list of inputs for the RNN inner loop
  _X = tf.split(0, n_steps, _X) # n_steps * (batch_size, n_hidden)

  # Get lstm cell output
  outputs = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, _X,
                                  initial_state_fw=_istate_fw,
                                  initial_state_bw=_istate_bw,
                                  sequence_length=_seq_len)

  # Linear activation
  # Get inner loop last output
  return tf.matmul(outputs[-1], _weights['out']) + _biases['out']

pred = BiRNN(x, istate_fw, istate_bw, weights, biases, BATCH_SIZE, n_steps)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost) # Adam Optimizer

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as s:
  start_time = time.time()
  s.run(init)
  batch_size = BATCH_SIZE
  display_step = DISPLAY_STEP
  num_epochs = NUM_EPOCHS

  train_data, train_profile_ids, train_labels = \
    extract_data(TRAIN_DATA_PATH, NUM_FEATURES, NUM_CLASSES,
                 IMAGE_HEIGHT, IMAGE_WIDTH,
                 NUM_INPUT_CHANNELS)

  valid_size = int(len(train_labels) * VALIDATION_DATA_RATIO)
  valid_data = train_data[:valid_size, :, :, :]
  valid_labels = train_labels[:valid_size]

  train_data = train_data[valid_size:, :, :, :]
  train_labels = train_labels[valid_size:]
  train_size = len(train_labels)

  train_data_shape = train_data.shape
  valid_data_shape = valid_data.shape
  # train_data: 4D array: [num_instances, image_height,
  #                       [num_features*image_width, num_input_channels]
  # Reshape and transpose data to get 28 sequences
  train_data = train_data.reshape(
    (train_data_shape[0], train_data_shape[1], train_data_shape[2])
  )
  valid_data = valid_data.reshape(
    (valid_data_shape[0], valid_data_shape[1], valid_data_shape[2])
  )

  train_xs = np.transpose(train_data, (0, 2, 1))
  # batch_xs: 3D array: [num_instances, image_width, image_height]
  train_ys = train_labels

  valid_xs = np.transpose(valid_data, (0, 2, 1))
  valid_ys = valid_labels

  validation_frequency = min(train_size, VALIDATION_FREQUENCY)
  best_validation_acc = 0.0
  best_step = 0

  # Keep training until reach max iterations
  for step in xrange(num_epochs * train_size // batch_size):
    # Compute the offset of the current minibatch in the data.
    offset = (step * batch_size) % (train_size - batch_size)
    batch_xs = train_xs[offset:(offset+batch_size), :, :]
    batch_ys = train_ys[offset:(offset+batch_size)]

    train_feed_dict = {x: batch_xs, y: batch_ys,
                       istate_fw: np.zeros((batch_size, 2*n_hidden)),
                       istate_bw: np.zeros((batch_size, 2*n_hidden))}
    # Fit training using batch data
    s.run(optimizer, feed_dict=train_feed_dict)

    curr_epoch_in_float = float(step) * batch_size / train_size

    if step % display_step == 0:
      # Calculate batch accuracy
      acc = s.run(accuracy, feed_dict=train_feed_dict)
      # Calculate batch loss
      loss = s.run(cost, feed_dict=train_feed_dict)

      print("@ Epoch %.2f" % curr_epoch_in_float)
      print("Minibatch loss: %.6f" % loss)
      print("Minibatch error: %.5f" % (1.0 - acc))

    if step % validation_frequency == 0:
      valid_feed_dict = {x: valid_xs, y: valid_ys,
                         istate_fw: np.zeros((valid_size, 2*n_hidden)),
                         istate_bw: np.zeros((valid_size, 2*n_hidden))}
      this_validation_acc = s.run(accuracy, feed_dict=valid_feed_dict)
      print("   Validation error: %.5f" % (1.0 - this_validation_acc))

      # If we got the best validation score until now
      if this_validation_acc > best_validation_acc:
        # Save the best validation score and step number
        best_validation_acc = this_validation_acc
        best_step = step


  duration = time.time() - start_time
  print("Optimization Finished!")
  print("Best validation error: %.5f obtained at epoch %.2f"
        % (1.0 - best_validation_acc,
           float(best_step) * batch_size / train_size))
  print >> sys.stderr, ("The session ran for %.2fm" % (duration / 60.))

  # Calculate accuracy for test data
  test_data, test_profile_ids, test_labels =\
    extract_data(TEST_DATA_PATH, NUM_FEATURES, NUM_CLASSES,
                 IMAGE_HEIGHT, IMAGE_WIDTH,
                 NUM_INPUT_CHANNELS)
  num_instances, image_height, image_width, _ = test_data.shape
  # test_data: 4D array: [num_instances, image_height,
  #                       [num_features*image_width, num_input_channels]
  # Reshape and transpose data to get 28 sequences
  test_data = test_data.reshape((num_instances, image_height, image_width))
  test_xs = np.transpose(test_data, (0, 2, 1))
  # batch_xs: 3D array: [num_instances, image_width, image_height]
  test_ys = test_labels
  test_len = 1000
  test_ys = test_ys[:test_len, :]

  test_feed_dict = {x: test_xs, y: test_ys,
                    istate_fw: np.zeros((test_len, 2*n_hidden)),
                    istate_bw: np.zeros((test_len, 2*n_hidden))}
  print "Testing Accuracy: ", s.run(accuracy, feed_dict=test_feed_dict)






