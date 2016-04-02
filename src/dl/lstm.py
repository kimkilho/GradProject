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


class LSTM(object):
    def __init__(self, tag, num_classes, label_name,
                 input_layer_size,
                 conv1_filter_size,
                 conv2_filter_size,
                 conv3_filter_size,
                 num_lstm_cells,
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
        :param conv1_filter_size: 1D list[int]:
                                 (filter_height, filter_width,
                                  input_num_channels, conv1_num_channels)
        :param conv2_filter_size: 1D list[int]:
                                 (filter_height, filter_width,
                                  conv1_num_channels, conv2_num_channels)
        :param conv3_filter_size: 1D list[int]:
                                 (filter_height, filter_width,
                                  conv2_num_channels, conv3_num_channels)
        :param num_lstm_cells: int
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
        self.conv1_filter_size = conv1_filter_size
        self.conv2_filter_size = conv2_filter_size
        self.conv3_filter_size = conv3_filter_size
        self.num_lstm_cells = num_lstm_cells
        self.num_hiddens = num_hiddens
        self.forget_bias = forget_bias
        self.stddev = stddev
        self.seed = seed
        self.train_ckpt_dir = train_ckpt_dir

        self.initialize()

    def initialize(self):
        num_features = self.input_layer_size[1]

        # Convolutional layers
        self.conv_layers_dict = {}  # {layer_name: [weight, bias]}
        for conv_layer_name, conv_filter_size \
            in zip(["conv1", "conv2", "conv3"],
                   [self.conv1_filter_size, self.conv2_filter_size,
                    self.conv3_filter_size]):
            weight = tf.Variable(
                tf.random_normal(conv_filter_size,
                                 stddev=self.stddev, seed=self.seed),
                name="%s_weight" % conv_layer_name
            )
            bias = tf.Variable(
                tf.zeros([conv_filter_size[3]]),
                name="%s_bias" % conv_layer_name
            )
            self.conv_layers_dict[conv_layer_name] = [weight, bias]
        #
        # self.rnn_layers_dict = {}   # {layer_name: [weight, bias]}
        # rnn1_weight_shape = [num_features, 2*self.num_hiddens]
        # rnn2_weight_shape = [2*self.num_hiddens, self.num_classes]
        # # rnn_weight_shape: 3D Tensor: [num_channels,
        # for dense_layer_name, weight_shape \
        #     in zip(["dense1", "dense2"],
        #            [hidden_weight_shape, out_weight_shape]):
        #
        #     weight = tf.Variable(
        #         tf.random_normal(weight_shape,
        #                          stddev=self.stddev, seed=self.seed),
        #         name="%s_weight" % layer_name
        #     )
        #     bias = tf.Variable(
        #         tf.zeros([weight_shape[1]]),
        #         name="%s_bias" % layer_name
        #     )
        #     self.layers_dict[layer_name] = [weight, bias]

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_hiddens,
                                                 forget_bias=self.forget_bias)
        # Dense1 layer
        self.dense1_stacked_lstm = \
            tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.num_lstm_cells)
        # Dense2 layer
        self.dense2_stacked_lstm = \
            tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.num_lstm_cells)

        self.dense_layers_dict = {}     # {layer_name: [weight, bias]}
        softmax_weight_shape = [self.num_hiddens, self.num_classes]
        for dense_layer_name, weight_shape \
            in zip(["softmax"],
                   [softmax_weight_shape]):
            # Softmax layer
            weight = tf.Variable(
                tf.random_normal([self.num_hiddens, self.num_classes],
                                 stddev=self.stddev, seed=self.seed),
                name="%s_weight" % dense_layer_name
            )
            bias = tf.Variable(
                tf.zeros([self.num_classes]),
                name="%s_bias" % dense_layer_name
            )
            self.dense_layers_dict[dense_layer_name] = [weight, bias]

    def model(self, data, train=False):
        """ The Model definition. """
        # data: 4D Tensor [num_instances, num_timesteps(image_height),
        #                    num_features(image_width), input_num_channels]
        num_timesteps = self.input_layer_size[0]
        num_features = self.input_layer_size[1]

        data_shape = data.get_shape().as_list()

        # Convolutional layer activations
        conv_actives_dict = {}  # {layer_name: Tensor}
        conv_actives_dict["input"] = data
        for prev_layer_name, conv_layer_name, nonlinear_layer_name \
            in zip(["input", "nonlinear1", "nonlinear2"],
                   ["conv1", "conv2", "conv3"],
                   ["nonlinear1", "nonlinear2", "nonlinear3"]):
            prev_active = conv_actives_dict[prev_layer_name]
            weight, bias = self.conv_layers_dict[conv_layer_name]

            conv_active = tf.nn.bias_add(
                tf.nn.conv2d(
                    prev_active, weight,
                    [1, 1, 1, 1],
                    padding="VALID"
                ), bias
            )

            conv_actives_dict[conv_layer_name] = conv_active

            nonlinear_active = tf.nn.relu(conv_active)
            conv_actives_dict[nonlinear_layer_name] = nonlinear_active

        dense_actives_dict = {}     # {layer_name: Tensor}
        dense1_input = conv_actives_dict["nonlinear3"]
        curr_num_instances, curr_num_timesteps, _, _ = \
            dense1_input.get_shape().as_list()
        # dense1_input: 4D Tensor: [num_instances, num_timesteps-a,
        #                           num_features, conv3_num_channels]
        dense1_input = tf.reshape(dense1_input,
                                  [curr_num_instances, curr_num_timesteps, -1])
        # dense1_input: 3D Tensor: [curr_num_instances, curr_num_timesteps,
        #                           num_features*conv3_num_channels]
        # print("dense1_input.get_shape()", dense1_input.get_shape())

        dense1_outputs = []
        state = self.dense1_stacked_lstm.zero_state(curr_num_instances, tf.float32)
        with tf.variable_scope("Dense1"):
            for timestep in range(curr_num_timesteps):
                if timestep > 0: tf.get_variable_scope().reuse_variables()
                # The value of state is updated after processing each batch
                output, state = \
                    self.dense1_stacked_lstm(dense1_input[:, timestep, :], state)
                output = tf.reshape(output, [curr_num_instances, 1, -1])
                # output: 3D Tensor: [curr_num_instances, 1, num_hiddens]
                dense1_outputs.append(output)
        # print("dense1_outputs[0].get_shape()", dense1_outputs[0].get_shape())
        dense2_input = tf.concat(1, dense1_outputs)
        # dense2_input: 2D Tensor: [curr_num_instances, curr_num_timesteps,
        #                           num_hiddens]
        dense_actives_dict["dense1"] = dense2_input

        dense2_outputs = []
        state = self.dense2_stacked_lstm.zero_state(curr_num_instances, tf.float32)
        with tf.variable_scope("Dense2"):
            for timestep in range(curr_num_timesteps):
                if timestep > 0: tf.get_variable_scope().reuse_variables()
                # The value of state is updated after processing each batch
                output, state = \
                    self.dense2_stacked_lstm(dense2_input[:, timestep, :], state)
                # output: 2D Tensor: [curr_num_instances, num_hiddens]
                dense2_outputs.append(output)
        # print("dense2_outputs[0].get_shape()", dense2_outputs[0].get_shape())

        # Softmax layer
        softmax_input = dense2_outputs[-1]
        dense_actives_dict["dense2"] = softmax_input
        softmax_weight, softmax_bias = self.dense_layers_dict["softmax"]

        logits = tf.nn.bias_add(tf.matmul(softmax_input, softmax_weight),
                                softmax_bias)

        # FIXME:
        if train:
            return logits
        else:
            return logits, conv_actives_dict, dense_actives_dict

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

        with tf.variable_scope("model", reuse=None):
            logits = self.model(train_data_node, train=True)
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
            learning_rate=learning_rate).minimize(loss, global_step=batch)

        train_prediction = tf.nn.softmax(logits)
        with tf.variable_scope("model", reuse=True):
            valid_logits, _, _ = self.model(valid_data_node)
            valid_prediction = tf.nn.softmax(valid_logits)

        valid_frequency = min(train_size, patience)   # check every epoch
        best_valid_error = np.inf
        best_step = 0

        trained_model_save_dir = self.train_ckpt_dir

        with tf.Session() as s:
            start_time = time.time()
            # Create a saver.
            print [var.name for var in tf.trainable_variables()]
            # FIXME: Add LSTM related weights and biases
            variables_to_be_saved_list = tf.trainable_variables()
            # variables_to_be_saved_list = []
            # for layer_name in self.conv_layers_dict:
            #     weight, bias = self.conv_layers_dict[layer_name]
            #     variables_to_be_saved_list.append(weight)
            #     variables_to_be_saved_list.append(bias)
            # for layer_name in self.dense_layers_dict:
            #     weight, bias = self.dense_layers_dict[layer_name]
            #     variables_to_be_saved_list.append(weight)
            #     variables_to_be_saved_list.append(bias)

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

        with tf.variable_scope("model", reuse=None):
            test_logits, _, _ = self.model(test_data_node)
            test_prediction = tf.nn.softmax(test_logits)

        trained_model_save_dir = self.train_ckpt_dir

        with tf.Session() as s:
            tf.initialize_all_variables().run()

            # Create a saver.
            variables_to_be_restored_list = tf.trainable_variables()
            # variables_to_be_restored_list = []
            # for layer_name in self.conv_layers_dict:
            #     weight, bias = self.conv_layers_dict[layer_name]
            #     variables_to_be_restored_list.append(weight)
            #     variables_to_be_restored_list.append(bias)
            # for layer_name in self.dense_layers_dict:
            #     weight, bias = self.dense_layers_dict[layer_name]
            #     variables_to_be_restored_list.append(weight)
            #     variables_to_be_restored_list.append(bias)
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
            test_pred = test_prediction.eval()
            predicted_label_symbols = np.argmax(test_pred, 1)
            test_label_symbols = np.argmax(test_labels, 1)
            print(predicted_label_symbols.shape, test_labels.shape)

            test_error = error_rate(test_pred, test_labels)
            print("Final test error: %.1f%%" % test_error)
            print(confusion_matrix(test_pred, test_labels))

        ops.reset_default_graph()   # NOTE: reset existing graph

