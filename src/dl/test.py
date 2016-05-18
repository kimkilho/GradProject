import tensorflow as tf
import os
import numpy as np

from lstm import LSTM
from util.preprocess_data import FEATURE_IDXS_DICT
from train import extract_data, parse_args, NUM_CLASSES, STDDEV, SEED

FLAGS = tf.app.flags.FLAGS

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "model")


def main(argv=None):
    TAG, LABEL_NAME, LEAVE_ONE_PERSON_OUT, \
    NUM_EPOCHS, BATCH_SIZE, \
    NUM_TIMESTEPS, INPUT_NUM_CHANNELS, \
    CONV1_FILTER_HEIGHT, CONV1_NUM_CHANNELS, \
    CONV2_FILTER_HEIGHT, CONV2_NUM_CHANNELS, \
    CONV3_FILTER_HEIGHT, CONV3_NUM_CHANNELS, \
    NUM_HIDDENS, FORGET_BIAS, \
    LEARNING_RATE, DROPOUT_PROB = \
        parse_args(FLAGS)

    CONV1_FILTER_WIDTH = CONV2_FILTER_WIDTH = CONV3_FILTER_WIDTH = 1

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

    print "TAG, LABEL_NAME, LEAVE_ONE_PERSON_OUT"
    print TAG, LABEL_NAME, LEAVE_ONE_PERSON_OUT
    print "NUM_EPOCHS, BATCH_SIZE"
    print NUM_EPOCHS, BATCH_SIZE
    print "NUM_TIMESTEPS, NUM_FEATURES (input_height, input_width), INPUT_NUM_CHANNELS"
    print NUM_TIMESTEPS, NUM_FEATURES, INPUT_NUM_CHANNELS
    print "CONV1_FILTER_HEIGHT, CONV1_NUM_CHANNELS"
    print CONV1_FILTER_HEIGHT, CONV1_NUM_CHANNELS
    print "CONV2_FILTER_HEIGHT, CONV2_NUM_CHANNELS"
    print CONV2_FILTER_HEIGHT, CONV2_NUM_CHANNELS
    print "CONV3_FILTER_HEIGHT, CONV3_NUM_CHANNELS"
    print CONV3_FILTER_HEIGHT, CONV3_NUM_CHANNELS
    print "NUM_HIDDENS, FORGET_BIAS"
    print NUM_HIDDENS, FORGET_BIAS
    print "LEARNING_RATE, DROPOUT_PROB"
    print LEARNING_RATE, DROPOUT_PROB

    TEST_DATA_PATH = os.path.join(DATA_DIR,
                                  "integrated_data_%s_I%d_%s_test.dat" %
                                  (TAG, NUM_TIMESTEPS, LABEL_NAME))

    TRAIN_CKPT_DIR = \
        os.path.join(MODEL_DIR,
                     "train_%s_NT%d_NF%d_INC%d_"
                     "C1FH%d_C1NC%d_C2FH%d_C2NC%d_C3FH%d_C3NC%d_"
                     "NH%d_FB%.2f_"
                     "LR%.4f_DP_%.1f_%s_%s" %
                     (TAG,
                      NUM_TIMESTEPS, NUM_FEATURES, INPUT_NUM_CHANNELS,
                      CONV1_FILTER_HEIGHT, CONV1_NUM_CHANNELS,
                      CONV2_FILTER_HEIGHT, CONV2_NUM_CHANNELS,
                      CONV3_FILTER_HEIGHT, CONV3_NUM_CHANNELS,
                      NUM_HIDDENS, FORGET_BIAS,
                      LEARNING_RATE,
                      DROPOUT_PROB,
                      "LOPO" if LEAVE_ONE_PERSON_OUT else "WHOLE",
                      LABEL_NAME))
    print("TRAIN_CKPT_DIR", TRAIN_CKPT_DIR)

    # Get the data.
    test_data, test_profile_ids, test_labels = \
        extract_data(TEST_DATA_PATH, NUM_FEATURES, NUM_CLASSES,
                 INPUT_NUM_CHANNELS, NUM_TIMESTEPS)
    # test_data: A 4D tensor [num_instances, num_timesteps(image_height),
    #                         num_features(image_width), input_num_channels]

    if LEAVE_ONE_PERSON_OUT:
        unique_profile_ids = np.unique(test_profile_ids)

        for profile_id in unique_profile_ids:
            test_data_only_pid = test_data[test_profile_ids == profile_id]
            test_labels_only_pid = test_labels[test_profile_ids == profile_id]

            print("Testing the model as a leave-one-person-out: profile_id=%d" 
                  % profile_id)
            lstm = LSTM(TAG, NUM_CLASSES, LABEL_NAME,
                          [NUM_TIMESTEPS, NUM_FEATURES, INPUT_NUM_CHANNELS],
                          [CONV1_FILTER_HEIGHT, CONV1_FILTER_WIDTH,
                           INPUT_NUM_CHANNELS, CONV1_NUM_CHANNELS],
                          [CONV2_FILTER_HEIGHT, CONV2_FILTER_WIDTH,
                           CONV1_NUM_CHANNELS, CONV2_NUM_CHANNELS],
                          [CONV3_FILTER_HEIGHT, CONV3_FILTER_WIDTH,
                           CONV2_NUM_CHANNELS, CONV3_NUM_CHANNELS],
                          NUM_HIDDENS,
                          FORGET_BIAS,
                          DROPOUT_PROB,
                          STDDEV, SEED,
                          TRAIN_CKPT_DIR,
                          profile_id=profile_id)
            lstm.test(test_data_only_pid, test_labels_only_pid)

    else:
        print("Testing the model for the whole data set...")
        with tf.device("/gpu:1"):
            lstm = LSTM(TAG, NUM_CLASSES, LABEL_NAME,
                          [NUM_TIMESTEPS, NUM_FEATURES, INPUT_NUM_CHANNELS],
                          [CONV1_FILTER_HEIGHT, CONV1_FILTER_WIDTH,
                           INPUT_NUM_CHANNELS, CONV1_NUM_CHANNELS],
                          [CONV2_FILTER_HEIGHT, CONV2_FILTER_WIDTH,
                           CONV1_NUM_CHANNELS, CONV2_NUM_CHANNELS],
                          [CONV3_FILTER_HEIGHT, CONV3_FILTER_WIDTH,
                           CONV2_NUM_CHANNELS, CONV3_NUM_CHANNELS],
                          NUM_HIDDENS,
                          FORGET_BIAS,
                          DROPOUT_PROB,
                          STDDEV, SEED,
                          TRAIN_CKPT_DIR)
            lstm.test(test_data, test_labels)


if __name__ == "__main__":
    tf.app.run()
