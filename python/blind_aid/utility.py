########################################################################################################################
# Authors:  Dan Iorga, Georgios Rizos, Georgios Theodorakis, Johannes Wiebe, Thomas Uriot
#
# BlindAid: HiPEDS CDT group project - cohort 2017 - Imperial College London
########################################################################################################################

########################################################################################################################
# Some common utility functions.
########################################################################################################################

import sys
import os
import inspect

from python import blind_aid


def get_package_path():
    """
    Returns the folder path that the package lies in.
    :return: folder_path: The package folder path.
    """
    return os.path.dirname(inspect.getfile(blind_aid))


def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


def get_column_names():
    column_names = ["checkpoint",
                    "radio_1",
                    "radio_2",
                    "radio_3",
                    "radio_4",
                    "sonar_1",
                    "sonar_2",
                    "sonar_3",
                    "sonar_4",
                    "acc_1",
                    "acc_2",
                    "acc_3",
                    "magnetic_1",
                    "magnetic_2",
                    "magnetic_3",
                    "compass",
                    "timestamp"]
    return column_names


def get_radio_features_names():
    features_names = ["radio_1",
                      "radio_2",
                      "radio_3",
                      "radio_4"]

    return features_names


def get_radio_compass_features_names():
    features_names = ["radio_1",
                      "radio_2",
                      "radio_3",
                      "radio_4",
                      "magnetic_1",
                      "magnetic_2",
                      "magnetic_3",
                      "compass_x",
                      "compass_y"]

    return features_names


def get_all_features_names():
    features_names = ["radio_1",
                      "radio_2",
                      "radio_3",
                      "radio_4",
                      "sonar_1",
                      "sonar_2",
                      "sonar_3",
                      "sonar_4",
                      "magnetic_1",
                      "magnetic_2",
                      "magnetic_3",
                      "compass_x",
                      "compass_y"]

    return features_names


def get_no_compass_features_names():
    features_names = ["radio_1",
                      "radio_2",
                      "radio_3",
                      "radio_4",
                      "sonar_1",
                      "sonar_2",
                      "sonar_3",
                      "sonar_4"]

    return features_names
