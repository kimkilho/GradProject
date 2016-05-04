import tensorflow as tf
import os
import numpy as np

from bl.rnn_wo_cnn import RNN
from util.preprocess_data import FEATURE_IDXS_DICT
from bl.train_bl import parse_args, extract_data, NUM_CLASSES, STDDEV, SEED

FLAGS = tf.app.flags.FLAGS

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "model_bl")


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

    TEST_DATA_PATH = os.path.join(DATA_DIR,
                                  "integrated_data_%s_I%d_%s_test.dat" %
                                  (TAG, NUM_TIMESTEPS, LABEL_NAME))

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
    print("TRAIN_CKPT_DIR", TRAIN_CKPT_DIR)

    # Get the data.
    test_data, test_profile_ids, test_labels = \
        extract_data(TEST_DATA_PATH, NUM_FEATURES, NUM_CLASSES,
                     INPUT_NUM_CHANNELS, NUM_TIMESTEPS)
    # test_data: A 4D tensor [num_instances, num_timesteps(image_height),
    #                         num_features(image_width), input_num_channels]

    print("Testing the model...")
    rnn = RNN(TAG, NUM_CLASSES, LABEL_NAME,
              [NUM_TIMESTEPS, NUM_FEATURES, INPUT_NUM_CHANNELS],
              NUM_HIDDENS,
              DROPOUT_PROB,
              STDDEV, SEED,
              TRAIN_CKPT_DIR)
    rnn.test(test_data, test_labels)


if __name__ == "__main__":
    tf.app.run()
