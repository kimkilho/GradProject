import os.path
from os import listdir, makedirs
import numpy as np
from sklearn import preprocessing
import argparse

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "data")
DATA_DIR = os.path.join(BASE_DIR, "AMGOLPAR", "fillna")     # FIXME
VALID_TAGS = "AMGOLPaR"
VALID_LABELS = {"accompanying", "conversing"}
VALID_NORMALIZE_OPTIONS = {"each", "all"}
# DEFAULT_AGGREGATED_DATA_DIR = os.path.join(DEFAULT_RAW_DATA_DIR, "aggregated")

""" CSV related notations """
SEP_CHAR = ','
LINE_BREAK_CHAR = '\n'

""" Training/Validation/Test data set ratio """
TRAIN_SET_RATIO = 7
VALID_SET_RATIO = 1
TEST_SET_RATIO = 2
LIMIT_TEST_NUM_INSTANCES = 10000

""" Original data file indices """
# [profile_id[0], timestamp[1], expId[2], accX[3], accY[4], accZ[5],
#  magnX[6], magnY[7], magnZ[8], gyroX[9], gyroY[10], gyroZ[11],
#  roll[12], pitch[13], azimuth[14], lux[15], distance[16],
#  l1Norm[17], l2Norm[18], linfNorm[19],
#  mfcc1[20], mfcc2[21], mfcc3[22], mfcc4[23],
#  mfcc5[24], mfcc6[25], mfcc7[26], mfcc8[27],
#  mfcc9[28], mfcc10[29], mfcc11[30], mfcc12[31],
#  psd1[32], psd2[33], psd3[34], psd4[35],
#  cosThetaOver2[36], xSinThetaOver2[37], ySinThetaOver2[38], zSinThetaOver2[39],
#  accompanying[40], conversing[41], drinking[42], having_a_meal[43],
#  in_class[44], sleeping[45]]
PROFILE_ID_IDX = 0
TIMESTAMP_IDX = 1
FEATURE_IDXS_DICT = {
    "A": (3, 6),
    "M": (6, 9),
    "G": (9, 12),
    "O": (12, 15),
    "L": (15, 16),
    "P": (16, 17),
    "a": (17, 36),
    "R": (36, 40)
}    # (inclusive, exclusive)
FEATURE_NAMES_DICT = {
    "A": ("accX", "accY", "accZ",),
    "M": ("magnX", "magnY", "magnZ",),
    "G": ("gyroX", "gyroY", "gyroZ",),
    "O": ("roll", "pitch", "azimuth",),
    "L": ("lux",),
    "P": ("distance",),
    "a": ("l1Norm", "l2Norm", "linfNorm",
          "mfcc1", "mfcc2", "mfcc3", "mfcc4",
          "mfcc5", "mfcc6", "mfcc7", "mfcc8",
          "mfcc9", "mfcc10", "mfcc11", "mfcc12",
          "psd1", "psd2", "psd3", "psd4",),
    "R": ("cosThetaOver2", "xSinThetaOver2",
          "ySinThetaOver2", "zSinThetaOver2",)
}
LABEL_IDXS_DICT = {"accompanying": 40, "conversing": 41}


def parse_args():
    parser = argparse.ArgumentParser(description="Input parameters "
                                                 "for data preprocessing.")
    parser.add_argument("--filter_tag", dest="filter_tag",
                        help="TAG indicating which modalities to be considered "
                             "while filtering missing values. "
                             "(A combination of 'A', 'M', 'G', 'O', 'L', 'P', 'a')")
    parser.add_argument("--normalize", dest="normalize",
                        help="Flag indicating to normalize "
                             "each persons' data or all persons' data "
                             "('each' or 'all', default: None)")
    parser.add_argument("--target_tag", dest="target_tag",
                        help="TAG indicating which modalities to be considered "
                             "for generating integrated data sets. "
                             "(A combination of letters in FILTER_TAG)")
    parser.add_argument("--label_name", dest="label_name",
                        help="The name of label to be classified "
                             "('sleeping' or 'having_a_meal')")
    parser.add_argument("--input_dim", dest="input_dim",
                        help="The input dimension of data to be generated "
                             "(default: 64)")
    args = parser.parse_args()
    if args.filter_tag and all(tag in set(VALID_TAGS) for tag in args.filter_tag):
        filter_tag = args.filter_tag
    else:
        raise argparse.ArgumentTypeError("TAG should be a combination of the characters: "
                                         "%s" % ", ".join([t for t in VALID_TAGS]))

    if args.target_tag and all(tag in set(filter_tag) for tag in args.target_tag):
        target_tag = args.target_tag
    else:
        raise argparse.ArgumentTypeError("TAG should be a combination of the characters: "
                                         "%s" % ", ".join([t for t in filter_tag]))

    if args.label_name and args.label_name in VALID_LABELS:
        label_name = args.label_name
    else:
        raise argparse.ArgumentTypeError("LABEL_TYPE should be one of the follows: "
                                         "sleeping, having_a_meal")

    if args.input_dim:
      input_dim = int(args.input_dim)
    else:
      input_dim = 64

    if args.normalize:
        if args.normalize in VALID_NORMALIZE_OPTIONS:
            normalize = args.normalize
        else:
            raise argparse.ArgumentTypeError("NORMALIZE should be one of the follows: "
                                             "each, all")
    else:
        normalize = None

    return filter_tag, target_tag, label_name, input_dim, normalize


def file_len(file_path):
    with open(file_path, 'r') as rf:
        for i, l in enumerate(rf):
            pass
    return i + 1


def shuffle_csv_file(data_file_path, random_sequence_file_path=None):
    """
    Shuffles the rows of the file
    :param data_file_path:
    @param random_sequence_file_path:
    :return:
    """
    if random_sequence_file_path:
        if os.path.exists(random_sequence_file_path):
            # Load existing random sequence from file
            with open(random_sequence_file_path, 'r') as rf:
                random_sequence = rf.read().rstrip().split(SEP_CHAR)
                random_sequence = [float(rand_num)
                                   for rand_num in random_sequence]
        else:
            import random
            file_length = file_len(data_file_path)
            # Generate a new random sequence and save it to file
            random_sequence = [random.random() for _ in range(file_length)]
            with open(random_sequence_file_path, 'w') as wf:
                wf.write(
                    SEP_CHAR.join(
                        [str(rand_num) for rand_num in random_sequence]
                    )
                )

        print "Start shuffling integrated data file with the order of " \
              "given random sequence..."

    # If None in random_sequence_file_path in random way:
    else:
        import random
        file_length = file_len(data_file_path)
        random_sequence = [random.random() for _ in range(file_length)]

        print 'Start shuffling integrated data file in random way...'

    with open(data_file_path, 'r') as rf:
        data = [(rand_num, line) for rand_num, line
                in zip(random_sequence, rf)]
    data.sort()

    # remove the existing output file
    if os.path.exists(data_file_path):
        os.remove(data_file_path)
    with open(data_file_path, 'w') as wf:
        for _, line in data:
            wf.write(line)


def normalize_files(raw_data_dir, normalized_data_dir,
                    target_feature_idxs, normalize_each=False):
    """
    Generates new csv files by normalizing samples to N(0, 1).
    :param raw_data_dir: string
    :param normalized_data_dir: string
    :param target_feature_idxs: tuple
    :param normalize_each: boolean
    :return:
    """

    if not os.path.exists(normalized_data_dir):
        makedirs(normalized_data_dir)

    def fit_normalizer(data_set):
        """
        Generates normalizer for given features.
        :param data_set: 2D np.array, (num_instances, num_features)
        :return: StandardScaler
        """
        return preprocessing.StandardScaler().fit(data_set)

    def transform_instance(normalizer, data_instance):
        """
        Transforms the data instance with pre-fitted normalizer.
        :param normalizer: StandardScaler
        :param data_instance: list (num_features)
        :return: 1D np.array (num_features)
        """
        # Applied simple trick to suppress deprecation warnings
        return normalizer.transform(data_instance.reshape(1, -1))[0]

    def normalize_file(normalizer, data_file, output_file,
                       target_feature_idxs, has_header=True):

        usecols = []
        for j in range(len(target_feature_idxs)):
            if j % 2 == 0:
                usecols += [i for i in range(target_feature_idxs[j],
                                             target_feature_idxs[j+1])]
        with open(output_file, 'w') as wf:
            with open(data_file, 'r') as rf:
                for i, line in enumerate(rf):
                    if has_header:
                        wf.write(line)
                        has_header = False
                        continue

                    line_list = line.rstrip().split(SEP_CHAR)

                    # Extract only feature values from the original line_list
                    feature_inst = []
                    for j in range(len(target_feature_idxs)):
                        if j % 2 == 0:
                            feature_inst += \
                                line_list[target_feature_idxs[j]: \
                                          target_feature_idxs[j+1]]

                    transformed_feature_inst = \
                        transform_instance(normalizer,
                                           np.asarray(feature_inst, dtype=float))

                    # Convert the original feature values
                    # to the normalized one
                    new_feature_inst = ["%.8f" % f
                                        for f in transformed_feature_inst]

                    for f in range(len(usecols)):
                        line_list[usecols[f]] = new_feature_inst[f]

                    wf.write(SEP_CHAR.join(line_list) + LINE_BREAK_CHAR)

    files_in_dir = []
    for f in listdir(raw_data_dir):
        if f.split('.')[-1] == "csv":
            files_in_dir.append(f)

    usecols = []
    for j in range(len(target_feature_idxs)):
        if j % 2 == 0:
            usecols += [i for i in range(target_feature_idxs[j],
                                         target_feature_idxs[j+1])]

    if not normalize_each:
        print 'Start fitting normalizer for all persons...'

        data_set_generated = False
        whole_feature_data_set = None
        for data_file in files_in_dir:
            data_file_path = os.path.join(raw_data_dir, data_file)
            curr_feature_data_set = \
                np.genfromtxt(data_file_path, delimiter=SEP_CHAR,
                              skip_header=1, usecols=usecols)

            # Filter the empty data sets
            if curr_feature_data_set.shape[0] == 0:
                print "Skipped an empty data set"
                continue

            if data_set_generated:
                whole_feature_data_set = \
                    np.append(whole_feature_data_set, curr_feature_data_set,
                              axis=0)
            else:
                whole_feature_data_set = curr_feature_data_set
                data_set_generated = True

            print "# rows: %d" % whole_feature_data_set.shape[0]

        normalizer = fit_normalizer(whole_feature_data_set)

    for data_file in files_in_dir:
        data_file_path = os.path.join(raw_data_dir, data_file)
        output_file_name = '.'.join(data_file.split('.')[:-1]) + "_normalized.csv"
        output_file_path = os.path.join(normalized_data_dir, output_file_name)
        print output_file_path

        if normalize_each:
            print 'Start fitting normalizer for a single person...'

            curr_feature_data_set = \
                np.genfromtxt(data_file_path, delimiter=SEP_CHAR,
                              skip_header=1, usecols=usecols)

            # Filter the empty data sets
            if curr_feature_data_set.shape[0] == 0:
                print "Skipped an empty data set"
                continue

            print "# rows: %d" % curr_feature_data_set.shape[0]
            normalizer = fit_normalizer(curr_feature_data_set)

        normalize_file(normalizer, data_file_path, output_file_path,
                       target_feature_idxs)


def filter_missing_values_in_files(raw_data_dir, filtered_data_dir,
                                   target_feature_idxs):
    """
    Generates new csv files by filtering samples
    with missing values in each csv files
    :param raw_data_dir: string
    :param filtered_data_dir: string
    :param target_feature_idxs: tuple
    :return:
    """

    if not os.path.exists(filtered_data_dir):
        makedirs(filtered_data_dir)

    def filter_missing_value_in_file(data_file, output_file, target_feature_idxs,
                                     has_header=True):
        with open(output_file, 'w') as wf:
            with open(data_file, 'r') as rf:
                for i, line in enumerate(rf):
                    if has_header:
                        wf.write(line)
                        has_header = False
                        continue

                    line_list = line.rstrip().split(SEP_CHAR)

                    # Extract only feature values from the original line_list
                    feature_inst = []
                    for j in range(len(target_feature_idxs)):
                        if j % 2 == 0:
                            feature_inst += \
                                line_list[target_feature_idxs[j]:\
                                          target_feature_idxs[j+1]]

                    # If feature_inst contains an empty string, pass
                    if '' in feature_inst:
                        continue

                    wf.write(line)

    files_in_dir = []
    for f in listdir(raw_data_dir):
        if f.split('.')[-1] == "csv":
            files_in_dir.append(f)

    for data_file in files_in_dir:
        data_file_path = os.path.join(raw_data_dir, data_file)
        output_file_name = '.'.join(data_file.split('.')[:-1]) + "_filtered.csv"
        output_file_path = os.path.join(filtered_data_dir, output_file_name)
        print(data_file_path)

        filter_missing_value_in_file(data_file_path, output_file_path,
                                     target_feature_idxs)


# DEPRECATED:
def aggregate_files(raw_data_dir, aggregated_data_dir, aggregation_frequency,
                    timestamp_idx):
    """
    Aggregate samples in each csv files with given frequency(Hz)
    and generates new csv files
    :param data_dir: string
    :param aggregation_frequency: int
    :return:
    """

    if not os.path.exists(aggregated_data_dir):
        makedirs(aggregated_data_dir)

    def aggregate_file(data_file, output_file,
                       timestamp_idx,
                       aggregation_frequency,
                       has_header=True):
        data_timestamp_diff = 1.0/aggregation_frequency   # sec (=1000ms)

        with open(output_file, 'w') as wf:
            with open(data_file, 'r') as rf:

                # Lists of original samples for a single aggregated sample
                data_orig_samples = []
                data_timestamp_orig_samples = []

                for i, line in enumerate(rf):
                    if has_header:
                        wf.write(line)
                        has_header = False
                        continue

                    line_list = line.rstrip().split(SEP_CHAR)
                    data_timestamp_sample = line_list[timestamp_idx]
                    data_orig_samples.append(line)
                    data_timestamp_orig_samples.append(data_timestamp_sample)

                    # if current sample's timestamp differs from
                    # the first sample's timestamp in the list
                    # by (1/aggregation_frequency) sec:
                    curr_data_timestamp_diff = \
                        abs(float(data_timestamp_orig_samples[0]) -
                            float(data_timestamp_sample))
                    if curr_data_timestamp_diff > data_timestamp_diff:
                        wf.write(data_orig_samples[0])

                        data_orig_samples = []
                        data_timestamp_orig_samples = []
                        data_orig_samples.append(line)
                        data_timestamp_orig_samples.append(data_timestamp_sample)

    files_in_dir = []
    for f in listdir(raw_data_dir):
        if f.split('.')[-1] == "csv":
            files_in_dir.append(f)

    for data_file in files_in_dir:
        data_file_path = os.path.join(raw_data_dir, data_file)
        output_file_name = '.'.join(data_file.split('.')[:-1]) + \
                           ("_aggregated_%dHz.csv" % aggregation_frequency)
        output_file_path = os.path.join(aggregated_data_dir, output_file_name)
        print(data_file_path)

        aggregate_file(data_file_path, output_file_path, timestamp_idx,
                       aggregation_frequency)


def integrate_to_file(raw_data_dir, integrated_file_path,
                      profile_id_idx, timestamp_idx,
                      feature_idxs, label_idx, num_features,
                      interval=64, overlap=0,
                      shuffle=False,
                      random_sequence_file_path=None):
    # TODO: Update description below
    """
    Runs integration of all csv files into one file
    The resulting file will be used by SSC-CNN models

    :param raw_data_dir: string, Raw data directory path
                         which includes files to be integrated
    :param integrated_file_path: string, Integrated data file path
    :param interval: int, Indicates how many raw instances to be integrated
                     into one
    :param overlap: int, Indicates how many raw instances to be overlapped
                    between two successive integrated instances
    :param shuffle: boolean, Indicates whether perform shuffling or not
    :return: None
    """

    def read_write_file(raw_data_file_path, integrated_file_path,
                        profile_id_idx, timestamp_idx,
                        feature_idxs, label_idx,
                        num_features, interval, overlap, has_header=True):
        count_timestamp_jumping = 0
        with open(integrated_file_path, 'a') as wf:  # Append to the existing file
            with open(raw_data_file_path, 'r') as rf:
                idx = 0
                profile_id_instances = []
                timestamp_instances = []
                feature_instances = []
                label_instances = []
                for i, line in enumerate(rf):
                    # Pass header row
                    if has_header:
                        has_header = False
                        continue

                    line_list = line.rstrip().split(SEP_CHAR)
                    profile_id_inst = line_list[profile_id_idx]
                    timestamp_inst = float(line_list[timestamp_idx])
                    feature_inst = []
                    for j in range(len(feature_idxs)):
                        if j % 2 == 0:
                            feature_inst += \
                                line_list[feature_idxs[j]:feature_idxs[j+1]]

                    # If feature_inst contains an empty string, pass
                    if '' in feature_inst:
                        continue

                    label_inst = str(int(float(line_list[label_idx])))
                    # FIXME: Pass labels 0, convert labels 1 to be 0
                    #        and regard labels over 1 to be 1 (not alone)
                    if label_idx == LABEL_IDXS_DICT["accompanying"] or \
                       label_idx == LABEL_IDXS_DICT["conversing"]:
                      if   label_inst == '0': continue
                      elif label_inst == '1': label_inst = '0'
                      else:                   label_inst = '1'
                    profile_id_instances.append(profile_id_inst)
                    timestamp_instances.append(timestamp_inst)
                    feature_instances.append(feature_inst)
                    label_instances.append(label_inst)
                    idx += 1

                    # if data instances list length has reached the interval length:
                    if idx % interval == 0:
                        # Put the very first timestamp value
                        # for single 3*interval instance
                        timestamp_instances_first = timestamp_instances[0]
                        feature_instances_reshaped = \
                            np.reshape(np.array(feature_instances),
                                       num_features*interval, order='F')

                        # Conditions for instance generation
                        # 1. All original raw instances in a new instance
                        #    must have the same value of label
                        # 2. Filter problematic data whose timestamp suddenly
                        #    jumps over (interval*avg_timestamp_diff_wo_max) sec
                        timestamp_instances_array = \
                            np.asarray(timestamp_instances, dtype=np.float64)
                        timestamp_diffs = np.diff(timestamp_instances_array)
                        idx_mask = np.ones(timestamp_diffs.shape, dtype=bool)
                        idx_mask[np.argmax(timestamp_diffs)] = False
                        max_timestamp_diff = np.max(timestamp_diffs)
                        avg_timestamp_diff_wo_max = np.mean(timestamp_diffs[idx_mask])

                        timestamp_jumping_cases_count = \
                            max_timestamp_diff > interval*avg_timestamp_diff_wo_max

                        write_condition = len(set(label_instances)) == 1 and \
                                          timestamp_jumping_cases_count == 0
                        if write_condition:
                            profile_id_inst = profile_id_instances[0]
                            label_inst = label_instances[0]
                            # Write current set of instances
                            wf.write(profile_id_inst + SEP_CHAR +
                                     str(timestamp_instances_first) + SEP_CHAR +
                                     SEP_CHAR.join(feature_instances_reshaped) +
                                     SEP_CHAR + label_inst + LINE_BREAK_CHAR)

                        elif len(set(label_instances)) != 1:
                            print("label mismatch", set(label_instances))
                        elif timestamp_jumping_cases_count != 0:
                            count_timestamp_jumping += 1
                            print("timestamp jumping: %d" % count_timestamp_jumping,
                                  timestamp_jumping_cases_count)
                            # print data_timestamp_diffs

                        # Re-initialize variables
                        if overlap > 0:
                            idx = overlap
                            profile_id_instances = \
                                profile_id_instances[-overlap:]
                            timestamp_instances = \
                                timestamp_instances[-overlap:]
                            feature_instances = feature_instances[-overlap:]
                            label_instances = label_instances[-overlap:]
                        else:
                            idx = 0
                            profile_id_instances = []
                            timestamp_instances = []
                            feature_instances = []
                            label_instances = []

    # remove the existing integrated file
    if os.path.exists(integrated_file_path):
        os.remove(integrated_file_path)
    files_in_dir = []
    for f in listdir(raw_data_dir):
        if f.split('.')[-1] == "csv":
            files_in_dir.append(f)

    for raw_data_filename in files_in_dir:
        raw_data_file_path = os.path.join(raw_data_dir, raw_data_filename)
        print(raw_data_file_path)
        read_write_file(raw_data_file_path, integrated_file_path,
                        profile_id_idx, timestamp_idx,
                        feature_idxs, label_idx, num_features,
                        interval, overlap)

    if shuffle:
        if random_sequence_file_path:
            shuffle_csv_file(integrated_file_path,
                             random_sequence_file_path=random_sequence_file_path)
        else:
            shuffle_csv_file(integrated_file_path)

    print("Integration complete")


def partition(data_file_path, train_file_path, test_file_path,
              train_ratio, valid_ratio, test_ratio):

    sum_ratio = train_ratio + valid_ratio + test_ratio
    total_num_inst = file_len(data_file_path)
    train_valid_num_inst = total_num_inst*(train_ratio+valid_ratio)/sum_ratio
    test_num_inst = total_num_inst - train_valid_num_inst
    print("train_valid_num_inst=%d" % train_valid_num_inst)
    print("test_num_inst=%d" % test_num_inst)

    wf_train = open(train_file_path, 'w')
    wf_test = open(test_file_path, 'w')
    with open(data_file_path, 'r') as rf:
        for i, line in enumerate(rf):
            if i <= train_valid_num_inst:
                wf_train.write(line)
            else:
                wf_test.write(line)

    wf_train.close()
    wf_test.close()
    print("Partitioning complete")


if __name__ == "__main__":
    FILTER_TAG, TAG, LABEL_NAME, INPUT_DIM, NORMALIZE = parse_args()

    INTEGRATED_DATA_PATH = os.path.join(BASE_DIR,
                                        "integrated_data_%s_I%d_%s.dat" %
                                        (TAG, INPUT_DIM, LABEL_NAME))

    TRAIN_DATA_PATH = os.path.join(BASE_DIR, "integrated_data_%s_I%d_%s_train.dat" %
                                   (TAG, INPUT_DIM, LABEL_NAME))
    TEST_DATA_PATH = os.path.join(BASE_DIR, "integrated_data_%s_I%d_%s_test.dat" %
                                  (TAG, INPUT_DIM, LABEL_NAME))

    # List of feature indices for filtering
    FILTER_FEATURE_IDXS = []
    for s in FILTER_TAG:
        FILTER_FEATURE_IDXS.append(FEATURE_IDXS_DICT[s])
    FILTER_FEATURE_IDXS = list(sum(FILTER_FEATURE_IDXS, ()))  # flatten the list

    # List of feature indices for normalizing
    ALL_FEATURE_IDXS = []
    for s in VALID_TAGS:
        ALL_FEATURE_IDXS.append(FEATURE_IDXS_DICT[s])
    ALL_FEATURE_IDXS = list(sum(ALL_FEATURE_IDXS, ()))  # flatten the list

    # List of feature indices for feature handling
    TARGET_FEATURE_IDXS = []
    for s in TAG:   # investigate TAG, character by character
        TARGET_FEATURE_IDXS.append(FEATURE_IDXS_DICT[s])
    FEATURE_IDXS = list(sum(TARGET_FEATURE_IDXS, ()))  # flatten the list
    NUM_FEATURES = 0
    for i in range(len(FEATURE_IDXS)):
        if i % 2 == 0:
            NUM_FEATURES -= FEATURE_IDXS[i]
        else:
            NUM_FEATURES += FEATURE_IDXS[i]
    LABEL_IDX = LABEL_IDXS_DICT[LABEL_NAME]

    """ Integrated data file indices """
    # [profile_id, timestamp, accX(64), accY(64), accZ(64),
    #  magnX(64), magnY(64), magnZ(64), gyroX(64), gyroY(64), gyroZ(64),
    #  roll(64), pitch(64), azimuth(64), lux(64), distance(64),
    #  l1Norm(64), l2Norm(64), linfNorm(64), label]
    INTEGRATED_PROFILE_ID_IDX = 0
    INTEGRATED_TIMESTAMP_IDX = 1
    INTEGRATED_LABEL_IDX = -1

    """ Data aggregation frequency """
    # AGGREGATION_FREQUENCY = 50

    """ Dimension and overlap for a single instance for DL """
    INPUT_OVERLAP = INPUT_DIM/2  # default=0
    SHUFFLE = True  # FIXME
    RANDOM_SEQUENCE_FILE_PATH = \
        os.path.join(BASE_DIR, "random_sequence_%s_%s.csv" %
                               (TAG, "20160323"))  # FIXME

    """ Filtering and normalizing of instances with missing values """
    FILTERED_DATA_DIR = os.path.join(DATA_DIR, "filtered_%s" % FILTER_TAG)
    FILTER_MISSING_VALUES = True    # FIXME
    NORMALIZED_DATA_DIR = os.path.join(DATA_DIR, "normalized_%s_%s" % (FILTER_TAG, NORMALIZE))

    if FILTER_MISSING_VALUES:
        if not os.path.exists(FILTERED_DATA_DIR):
            filter_missing_values_in_files(DATA_DIR, FILTERED_DATA_DIR,
                                           FILTER_FEATURE_IDXS)
        DATA_DIR = FILTERED_DATA_DIR

    # TODO: Make clear which features to be normalized
    if NORMALIZE:
        if not os.path.exists(NORMALIZED_DATA_DIR):
            if NORMALIZE == "each":
                normalize_files(DATA_DIR, NORMALIZED_DATA_DIR,
                                FILTER_FEATURE_IDXS, normalize_each=True)
            elif NORMALIZE == "all":
                normalize_files(DATA_DIR, NORMALIZED_DATA_DIR,
                                FILTER_FEATURE_IDXS)

        DATA_DIR = NORMALIZED_DATA_DIR

    # if AGGREGATION_FREQUENCY:
    #     aggregate_files(DATA_DIR, DEFAULT_AGGREGATED_DATA_DIR,
    #                     AGGREGATION_FREQUENCY, TIMESTAMP_IDX)
    # DATA_DIR = DEFAULT_AGGREGATED_DATA_DIR

    integrate_to_file(DATA_DIR, INTEGRATED_DATA_PATH,
                      PROFILE_ID_IDX, TIMESTAMP_IDX,
                      FEATURE_IDXS, LABEL_IDX, NUM_FEATURES,
                      interval=INPUT_DIM, overlap=INPUT_OVERLAP,
                      shuffle=SHUFFLE,
                      random_sequence_file_path=RANDOM_SEQUENCE_FILE_PATH)

    partition(INTEGRATED_DATA_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH,
              TRAIN_SET_RATIO, VALID_SET_RATIO, TEST_SET_RATIO)
