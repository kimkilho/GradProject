import tensorflow as tf
import os
import numpy as np

from bl.rnn_wo_cnn import RNN
from util.preprocess_data import TRAIN_SET_RATIO, VALID_SET_RATIO, \
    FEATURE_IDXS_DICT

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
    "num_epochs", 300,
    "The number of epochs "
    "(default: 300)")
tf.app.flags.DEFINE_integer(
    "batch_size", 128,
    "The batch size for training "
    "(default: 128)")

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
    "learning_rate", 0.01,
    "Learning rate of SGD (default: 0.01).")
tf.app.flags.DEFINE_float(
    "dropout_prob", 0.5,
    "Dropout probability of the model (default: 0.5)."
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "model_bl")

# Basic parameters
NUM_CLASSES = 2

# Initialization parameters
STDDEV = 0.1

# Learning algorithm parameters
VALIDATION_DATA_RATIO = float(VALID_SET_RATIO) / (TRAIN_SET_RATIO + VALID_SET_RATIO)
PATIENCE = 1000
SEED = None

LEARNING_RATE_DECAY_FACTOR = 0.99


def parse_args(flags):
    tag = flags.tag
    label_name = flags.label_name
    num_epochs = flags.num_epochs
    batch_size = flags.batch_size

    num_timesteps = flags.num_timesteps
    input_num_channels = flags.input_num_channels

    num_hiddens = flags.num_hiddens

    learning_rate = flags.learning_rate
    dropout_prob = flags.dropout_prob

    return tag, label_name, num_epochs, batch_size, \
           num_timesteps, input_num_channels, \
           num_hiddens, \
           learning_rate, dropout_prob


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
    TAG, LABEL_NAME, NUM_EPOCHS, BATCH_SIZE, \
    NUM_TIMESTEPS, INPUT_NUM_CHANNELS, \
    NUM_HIDDENS, \
    LEARNING_RATE, DROPOUT_PROB = \
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

    print "TAG, LABEL_NAME, NUM_EPOCHS, BATCH_SIZE"
    print TAG, LABEL_NAME, NUM_EPOCHS, BATCH_SIZE
    print "NUM_TIMESTEPS, NUM_FEATURES (input_height, input_width), INPUT_NUM_CHANNELS"
    print NUM_TIMESTEPS, NUM_FEATURES, INPUT_NUM_CHANNELS
    print "CONV1_FILTER_HEIGHT, CONV1_NUM_CHANNELS"
    print "NUM_HIDDENS"
    print NUM_HIDDENS
    print "LEARNING_RATE, DROPOUT_PROB"
    print LEARNING_RATE, DROPOUT_PROB

    TRAIN_DATA_PATH = os.path.join(DATA_DIR,
                                   "integrated_data_%s_I%d_%s_train.dat" %
                                   (TAG, NUM_TIMESTEPS, LABEL_NAME))
    # TRAIN_DATA_PATH = os.path.join(DATA_DIR, "sample.dat")

    TRAIN_CKPT_DIR = \
        os.path.join(MODEL_DIR,
                     "train_%s_NT%d_NF%d_INC%d_"
                     "NH%d_"
                     "LR%.4f_DP_%.1f_%s" %
                     (TAG,
                      NUM_TIMESTEPS, NUM_FEATURES, INPUT_NUM_CHANNELS,
                      NUM_HIDDENS,
                      LEARNING_RATE,
                      DROPOUT_PROB,
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
    rnn = RNN(TAG, NUM_CLASSES, LABEL_NAME,
               [NUM_TIMESTEPS, NUM_FEATURES, INPUT_NUM_CHANNELS],
               NUM_HIDDENS,
               DROPOUT_PROB,
               STDDEV, SEED,
               TRAIN_CKPT_DIR)
    rnn.train(LEARNING_RATE,
               BATCH_SIZE, NUM_EPOCHS,
               LEARNING_RATE_DECAY_FACTOR, PATIENCE,
               train_data, train_labels,
               valid_data, valid_labels)


if __name__ == "__main__":
    tf.app.run()
