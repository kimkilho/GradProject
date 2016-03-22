import tensorflow as tf
import os
import numpy as np

from blstm import BLSTM
from util.preprocess_data import TRAIN_SET_RATIO, VALID_SET_RATIO, \
    FEATURE_IDXS_DICT, FEATURE_NAMES_DICT

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    "tag", 'A',
    "TAG indicating which modalities to be considered "
    "(A combination of 'A', 'M', 'G', 'O', 'L', 'P', 'a', 'R', default: A).")
tf.app.flags.DEFINE_string(
    "label_name", "accompanying",
    "The name of label to be classified "
    "('accompanying' or 'conversing', default: accompanying).")
tf.app.flags.DEFINE_integer(
    "num_timesteps", 64,
    "The number of timesteps for input layer "
    "(default: 64).")
tf.app.flags.DEFINE_integer(
    "input_num_channels", 1,
    "The number of input channels "
    "(default: 1).")

tf.app.flags.DEFINE_integer(
    "num_hiddens", 64,
    "The number of hidden units in LSTM cell "
    "(default: 64).")
tf.app.flags.DEFINE_float(
    "forget_bias", 1.0,
    "Forget bias for LSTM cells "
    "(default: 1.0).")

tf.app.flags.DEFINE_float(
    "learning_rate", 0.01,
    "Learning rate of SGD (default: 0.01).")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "model")

# Basic parameters
NUM_CLASSES = 2

# Initialization parameters
STDDEV = 0.1

# Learning algorithm parameters
NUM_EPOCHS = 300
BATCH_SIZE = 128
VALIDATION_DATA_RATIO = float(VALID_SET_RATIO) / (TRAIN_SET_RATIO + VALID_SET_RATIO)
PATIENCE = 1000
SEED = None

LEARNING_RATE_DECAY_FACTOR = 0.99


def parse_args(flags):
    tag = flags.tag
    label_name = flags.label_name
    num_timesteps = flags.num_timesteps

    input_num_channels = flags.input_num_channels
    num_hiddens = flags.num_hiddens
    forget_bias = flags.forget_bias

    learning_rate = flags.learning_rate

    return tag, label_name, num_timesteps, \
           input_num_channels, num_hiddens, forget_bias, \
           learning_rate


def extract_data(filename, num_features, num_classes,
                 input_num_channels, num_timesteps,
                 num_instances=None):
    """
    Extract the instances and labels.
    :param filename: String
    :param num_classes : int
    :param num_timesteps: int
    :param num_features: int
    :param input_num_channels: int
    :param num_instances: int
    :return:
        data: A 4D tensor [num_instances, num_timesteps(image_height),
                           num_features(image_width), input_num_channels]
        labels: A 1-hot matrix [num_instances, NUM_CLASSES]
    """
    print("Extracting", filename)
    data_set = np.genfromtxt(filename, delimiter=',', dtype=np.float32)

    data = data_set[:, 2:-1]    # Exclude profile_id, timestamp, label
    # data: A 2D array: [num_instances, num_features*num_timesteps]
    profile_ids = data_set[:, 0]
    labels = data_set[:, -1]

    # Slice out the features and labels with the size of num_instances
    if not num_instances:
        num_instances = data.shape[0]
    data = data[:num_instances]
    profile_ids = profile_ids[:num_instances]
    labels = labels[:num_instances]

    data = data.reshape((-1, num_timesteps, num_features, 1), order='F')
    data = data.repeat(input_num_channels, axis=3)
    # data: A 4D array: [num_instances, num_timesteps,
    #                    num_features, input_num_channels]

    # Convert to dense 1-hot representation.
    labels = (np.arange(num_classes) == labels[:, None]).astype(np.float32)

    return data, profile_ids, labels


def main(argv=None):
    TAG, LABEL_NAME, NUM_TIMESTEPS, \
    INPUT_NUM_CHANNELS, NUM_HIDDENS, FORGET_BIAS, \
    LEARNING_RATE = \
        parse_args(FLAGS)

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

    print "TAG, LABEL_NAME"
    print TAG, LABEL_NAME
    print "NUM_TIMESTEPS, NUM_FEATURES (input_height, input_width)"
    print NUM_TIMESTEPS, NUM_FEATURES
    print "INPUT_NUM_CHANNELS, NUM_HIDDENS, FORGET_BIAS"
    print INPUT_NUM_CHANNELS, NUM_HIDDENS, FORGET_BIAS
    print "LEARNING_RATE"
    print LEARNING_RATE

    TRAIN_DATA_PATH = os.path.join(DATA_DIR,
                                   "integrated_data_%s_I%d_%s_train.dat" %
                                   (TAG, NUM_TIMESTEPS, LABEL_NAME))

    TRAIN_CKPT_DIR = \
        os.path.join(MODEL_DIR,
                     "train_%s_LR%.2f_NT%d_NF%d_INC%d_NH%d_%s" %
                     (TAG,
                      LEARNING_RATE,
                      NUM_TIMESTEPS, NUM_FEATURES,
                      INPUT_NUM_CHANNELS, NUM_HIDDENS,
                      LABEL_NAME))
    if not os.path.exists(TRAIN_CKPT_DIR):
        os.makedirs(TRAIN_CKPT_DIR)
    print("TRAIN_CKPT_DIR", TRAIN_CKPT_DIR)

    # Get the data.
    train_data, train_profile_ids, train_labels = \
        extract_data(TRAIN_DATA_PATH, NUM_FEATURES, NUM_CLASSES,
                     INPUT_NUM_CHANNELS, NUM_TIMESTEPS)
    # train_data: A 4D tensor [num_instances, num_timesteps(image_height),
    #                          num_features(image_width), input_num_channels]

    valid_size = int(train_labels.shape[0] * VALIDATION_DATA_RATIO)
    valid_data = train_data[:valid_size, :, :, :]
    valid_labels = train_labels[:valid_size]
    train_data = train_data[valid_size:, :, :, :]
    train_labels = train_labels[valid_size:]

    print("Training...")
    blstm = BLSTM(TAG, NUM_CLASSES, LABEL_NAME,
                  [NUM_TIMESTEPS, NUM_FEATURES, INPUT_NUM_CHANNELS],
                  NUM_HIDDENS,
                  FORGET_BIAS,
                  STDDEV, SEED,
                  TRAIN_CKPT_DIR)
    blstm.train(LEARNING_RATE,
                BATCH_SIZE, NUM_EPOCHS,
                LEARNING_RATE_DECAY_FACTOR, PATIENCE,
                train_data, train_labels,
                valid_data, valid_labels)


if __name__ == "__main__":
    tf.app.run()
