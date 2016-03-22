import tensorflow as tf
import os
import numpy as np

from blstm import BLSTM
from util.preprocess_data import FEATURE_IDXS_DICT
from train import extract_data, parse_args, NUM_CLASSES, STDDEV, SEED

FLAGS = tf.app.flags.FLAGS

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "model")


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

    TEST_DATA_PATH = os.path.join(DATA_DIR,
                                  "integrated_data_%s_I%d_%s_test.dat" %
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
    test_data, test_profile_ids, test_labels = \
        extract_data(TEST_DATA_PATH, NUM_FEATURES, NUM_CLASSES,
                 INPUT_NUM_CHANNELS, NUM_TIMESTEPS)
    # train_data: A 4D tensor [num_instances, num_timesteps(image_height),
    #                          num_features(image_width), input_num_channels]

    print("Testing the model...")
    blstm = BLSTM(TAG, NUM_CLASSES, LABEL_NAME,
                  [NUM_TIMESTEPS, NUM_FEATURES, INPUT_NUM_CHANNELS],
                  NUM_HIDDENS,
                  FORGET_BIAS,
                  STDDEV, SEED,
                  TRAIN_CKPT_DIR)
    blstm.test(test_data, test_labels)


if __name__ == "__main__":
    tf.app.run()
