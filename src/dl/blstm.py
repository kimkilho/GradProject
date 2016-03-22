import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
from tensorflow.python.ops.constant_op import constant
from tensorflow.python.framework import ops
from sklearn import metrics

from util.preprocess_data import FEATURE_IDXS_DICT, FEATURE_NAMES_DICT


def error_rate(predictions, labels):
    """
    Return the error rate based on dense predictions and 1-hot labels.
    :param predictions:
    :param labels:
    :return:
    """
    return 100 - (
        100.0 *
        np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
        predictions.shape[0]
    )


def confusion_matrix(predictions, labels):
    predicted_labels = np.argmax(predictions, 1)
    test_labels = np.argmax(labels, 1)

    return metrics.confusion_matrix(test_labels, predicted_labels)


class BLSTM(object):
    def __init__(self, tag, num_classes, label_name,
                 input_layer_size,
                 num_hiddens,
                 forget_bias,
                 stddev, seed,
                 train_ckpt_dir):
        """
        :param tag: String
        :param num_classes: int
        :param label_name: String
        :param input_layer_size: 1D list[int]:
                                 (num_timesteps, num_features, input_num_channels)
                                 = (input_height, input_width, input_num_channels)
        :param num_hiddens: int
        :param forget_bias: float
        :param stddev: float
        :param seed: int (or None)
        :param train_ckpt_dir: String
        :return: void
        """

        # Set GPU options
        # self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

        assert len(input_layer_size) == 3

        self.tag = tag
        self.num_classes = num_classes
        self.label_name = label_name
        self.input_layer_size = input_layer_size
        self.num_hiddens = num_hiddens
        self.forget_bias = forget_bias
        self.stddev = stddev
        self.seed = seed
        self.train_ckpt_dir = train_ckpt_dir

        self.initialize()

    def initialize(self):
        num_features = self.input_layer_size[1]

        self.layers_dict = {}   # {layer_name: [weight, bias]}

        hidden_weight_shape = [num_features, 2*self.num_hiddens]
        out_weight_shape = [2*self.num_hiddens, self.num_classes]
        for layer_name, weight_shape \
            in zip(["hidden", "out"],
                   [hidden_weight_shape, out_weight_shape]):

            weight = tf.Variable(
                tf.random_normal(weight_shape,
                                 stddev=self.stddev, seed=self.seed),
                name="%s_weight" % layer_name
            )
            bias = tf.Variable(
                tf.zeros([weight_shape[1]]),
                name="%s_bias" % layer_name
            )
            self.layers_dict[layer_name] = [weight, bias]

    def model(self, data, istate_fw, istate_bw):
        """ The Model definition. """
        # data: A 4D tensor [num_instances, num_timesteps(image_height),
        #                    num_features(image_width), input_num_channels]
        num_timesteps = self.input_layer_size[0]
        num_features = self.input_layer_size[1]

        # TODO: Add convolutional layers
        data_shape = data.get_shape().as_list()
        data = tf.reshape(data, [data_shape[0], data_shape[1], data_shape[2]])
        # data: A 3D tensor [num_instances, num_timesteps(image_height),
        #                    num_features(image_width)]

        _seq_len = tf.fill([data_shape[0]],
                           constant(num_timesteps, dtype=tf.int64))

        # data: A 3D Tensor: (batch_size, num_timesteps, num_features)
        data = tf.transpose(data, [1, 0, 2])    # permute num_timesteps and batch_size
        # Reshape to prepare input to hidden activation
        data = tf.reshape(data, [-1, num_features]) # (num_timesteps*batch_size, num_features)

        actives_dict = {}   # {layer_name: tf.Tensor}
        # Linear activation
        hidden_weight, hidden_bias = self.layers_dict["hidden"]
        actives_dict["hidden"] = \
            tf.nn.bias_add(tf.matmul(data, hidden_weight), hidden_bias)
        # actives_dict["hidden"]: A 2D Tensor: (num_timesteps*batch_size, 2*num_hiddens)

        # Define LSTM cells with TensorFlow
        # Forward direction cell
        lstm_fw_cell = rnn_cell.BasicLSTMCell(self.num_hiddens,
                                              forget_bias=self.forget_bias)
        lstm_bw_cell = rnn_cell.BasicLSTMCell(self.num_hiddens,
                                              forget_bias=self.forget_bias)
        # Split data because RNN Cell needs a list of inputs for the RNN inner loop
        rnn_input_list = tf.split(0, num_timesteps, actives_dict["hidden"])
        # rnn_input_list[0]: A 2D Tensor: (batch_size, 2*num_hiddens)

        # Get LSTM cell output
        rnn_output_list = \
            rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, rnn_input_list,
                                  initial_state_fw=istate_fw,
                                  initial_state_bw=istate_bw,
                                  sequence_length=_seq_len)
        # Linear activation
        # Get inner loop last output
        out_weight, out_bias = self.layers_dict["out"]
        # FIXME:
        return tf.nn.bias_add(tf.matmul(rnn_output_list[-1], out_weight),
                              out_bias)

    def train(self, base_learning_rate,
              batch_size, num_epochs,
              learning_rate_decay_factor, patience,
              train_data, train_labels,
              valid_data, valid_labels):
        num_timesteps = self.input_layer_size[0]
        num_features = self.input_layer_size[1]
        input_num_channels = self.input_layer_size[2]

        train_size = train_labels.shape[0]
        valid_size = valid_labels.shape[0]
        print("train_size", train_size)

        train_data_node = tf.placeholder(
            tf.float32,
            shape=(batch_size, num_timesteps, num_features, input_num_channels)
        )
        train_labels_node = \
            tf.placeholder(tf.float32, shape=(batch_size, self.num_classes))

        valid_data_node = tf.constant(valid_data)

        # FIXME:
        train_istate_fw = \
            tf.zeros([batch_size, 2*self.num_hiddens], dtype=tf.float32)
        train_istate_bw = \
            tf.zeros([batch_size, 2*self.num_hiddens], dtype=tf.float32)
        valid_istate_fw = \
            tf.zeros([valid_size, 2*self.num_hiddens], dtype=tf.float32)
        valid_istate_bw = \
            tf.zeros([valid_size, 2*self.num_hiddens], dtype=tf.float32)
        with tf.variable_scope("model_train", reuse=None):
            logits = self.model(train_data_node, train_istate_fw, train_istate_bw)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits, train_labels_node))

        # TODO: Add regularization term

        batch = tf.Variable(0, trainable=False)

        learning_rate = tf.train.exponential_decay(
            base_learning_rate,         # Base learning rate.
            batch * batch_size,         # Current index into the dataset.
            train_size,                 # Decay step.
            learning_rate_decay_factor, # Decay rate.
            staircase=True)

        # FIXME: Define loss and optimizer
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(loss)

        train_prediction = tf.nn.softmax(logits)
        with tf.variable_scope("model_valid", reuse=None):
            valid_prediction = tf.nn.softmax(
                self.model(valid_data_node, valid_istate_fw, valid_istate_bw))

        valid_frequency = min(train_size, patience)   # check every epoch
        best_valid_error = np.inf
        best_step = 0

        trained_model_save_dir = self.train_ckpt_dir

        with tf.Session() as s:
            start_time = time.time()
            # Create a saver.
            variables_to_be_saved_list = []
            for layer_name in self.layers_dict:
                weight, bias = self.layers_dict[layer_name]
                variables_to_be_saved_list.append(weight)
                variables_to_be_saved_list.append(bias)

            saver = tf.train.Saver(variables_to_be_saved_list)

            # Run all the initializers to prepare the trainable parameters.
            tf.initialize_all_variables().run()
            print("Variables initialized!")

            # Loop through training steps.
            for step in xrange(num_epochs * train_size // batch_size):
                # Compute the offset of the current minibatch in the data.
                # Note that we could use better randomization across epochs.
                offset = (step * batch_size) % (train_size - batch_size)
                batch_data = train_data[offset:(offset+batch_size), :, :, :]
                batch_labels = train_labels[offset:(offset+batch_size)]

                # This dictionary maps the batch data (as a numpy array)
                # to the node in the graph that should be fed to.
                feed_dict = {train_data_node: batch_data,
                             train_labels_node: batch_labels}
                # Run the graph and fetch some of the nodes.
                _, l, lr, predictions = s.run(
                    [optimizer, loss, learning_rate, train_prediction],
                    feed_dict=feed_dict)

                curr_epoch_in_float = float(step) * batch_size / train_size

                if step % 100 == 0:
                    print("@ Epoch %.2f" % curr_epoch_in_float)
                    print("Minibatch loss: %.3f, learning_rate: %.6f" % (l, lr))
                    print("Minibatch error: %.1f%%" %
                          error_rate(predictions, batch_labels))
                    sys.stdout.flush()

                if step % valid_frequency == 0:
                    this_valid_error = \
                        error_rate(valid_prediction.eval(), valid_labels)
                    print("    Validation error: %.1f%%" % this_valid_error)

                    # If we got the best validation score until now
                    if this_valid_error < best_valid_error:
                        # Save the best validation score and step number
                        best_valid_error = this_valid_error
                        best_step = step

                        # Save the model checkout periodically.
                        checkpoint_path = os.path.join(trained_model_save_dir,
                                                       "model.ckpt")
                        saver.save(s, checkpoint_path, global_step=step)

                    sys.stdout.flush()

                # Break loop early when the learning rate gets low enough
                if lr < 1e-7:
                    break

            duration = time.time() - start_time
            print("Optimization complete.")
            print("Best validation error: %.1f%% obtained at epoch %.2f" %
                  (best_valid_error, float(best_step) * batch_size / train_size))
            print >> sys.stderr, ("The session ran for %.2fm" % (duration / 60.))

        ops.reset_default_graph()   # NOTE: reset existing graph

    def test(self, test_data, test_labels):

        test_size = test_labels.shape[0]
        test_data_node = tf.constant(test_data)
        test_istate_fw = \
            tf.zeros([test_size, 2*self.num_hiddens], dtype=tf.float32)
        test_istate_bw = \
            tf.zeros([test_size, 2*self.num_hiddens], dtype=tf.float32)

        with tf.variable_scope("model_test", reuse=None):
            test_prediction = tf.nn.softmax(
                self.model(test_data_node, test_istate_fw, test_istate_bw))

        trained_model_save_dir = self.train_ckpt_dir

        with tf.Session() as s:
            # Create a saver.
            variables_to_be_restored_list = []
            for layer_name in self.layers_dict:
                weight, bias = self.layers_dict[layer_name]
                variables_to_be_restored_list.append(weight)
                variables_to_be_restored_list.append(bias)
            saver = tf.train.Saver(variables_to_be_restored_list)

            ckpt = tf.train.get_checkpoint_state(trained_model_save_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(s, ckpt.model_checkpoint_path)
                # Assuming model checkpoint path looks something like:
                #    /my-favorite-path/eval/model.ckpt-0.
                # extract global step from it.
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print("No checkpoint file found")
                return

            # Finally print the result
            predicted_label_symbols = np.argmax(test_prediction.eval(), 1)
            test_label_symbols = np.argmax(test_labels, 1)
            print(predicted_label_symbols.shape, test_labels.shape)

            test_error = error_rate(test_prediction.eval(), test_labels)
            print("Final test error: %.1f%%" % test_error)
            print(confusion_matrix(test_prediction.eval(), test_labels))

        ops.reset_default_graph()   # NOTE: reset existing graph

