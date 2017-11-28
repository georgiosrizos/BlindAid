########################################################################################################################
# Authors:  Dan Iorga, Georgios Rizos, Georgios Theodorakis, Johannes Wiebe, Thomas Uriot
#
# BlindAid: HiPEDS CDT group project - cohort 2017 - Imperial College London
########################################################################################################################

########################################################################################################################
# Some common functions used for data cleaning and preprocessing.
########################################################################################################################

import numpy as np
import pandas as pd
from scipy.signal import gaussian
from scipy.ndimage import filters


def read_csv_and_filter_missing(filepath, column_names):
    data_df = pd.read_csv(filepath, names=column_names, header=None)

    data_df = filter_values(data_df)

    data_df = data_df.replace(to_replace=-9999, value=np.nan)

    data_df = data_df.fillna(method="ffill", axis=0)
    data_df = data_df.replace(to_replace=np.nan, value=0)

    data_df = get_cartesian_compass(data_df)

    data = data_df.values

    return data_df, data


def filter_missing(data_s, column_names):
    data_df_s = pd.DataFrame(data_s, columns=column_names)

    data_df_s = filter_values(data_df_s)

    data_df_s = data_df_s.replace(to_replace=-9999, value=np.nan)

    data_df_s = data_df_s.fillna(method="ffill", axis=0)
    data_df_s = data_df_s.replace(to_replace=np.nan, value=0)

    data_df_s = get_cartesian_compass(data_df_s)

    data_s = data_df_s.values

    return data_df_s, data_s


def filter_values(data_df_s):
    data_s = data_df_s.values

    for j, column_name in enumerate(data_df_s.columns):
        if "sonar" in column_name:
            data_s[data_s[:, j] > 1000, j] = -9999  # This cutoff value was experimentally tuned.
            data_s[data_s[:, j] < 0, j] = -9999
        elif "radio" in column_name:
            data_s[data_s[:, j] > -420, j] = -9999
            data_s[data_s[:, j] < -1280, j] = -9999
        elif "compass" in column_name:
            data_s[data_s[:, j] > 3600, j] = -9999
            data_s[data_s[:, j] < 0, j] = -9999

    data_df_s = pd.DataFrame(data_s, columns=data_df_s.columns)
    return data_df_s


def get_cartesian_compass(data_df_s):
    data_s = data_df_s.values

    compass_cartesian = np.empty((data_s.shape[0], 2), dtype=np.float32)

    for i, d in enumerate(data_s[:, -2]):
        # radians = (d/10) * 0.0174533
        radians = np.radians(d / 10)
        x = np.sin(radians)
        y = np.cos(radians)
        compass_cartesian[i, 0] = x
        compass_cartesian[i, 1] = y

    data_s = np.hstack([data_s[:, :-1].astype(np.float32), compass_cartesian])
    data_df_s = pd.DataFrame(data_s, columns=list(data_df_s.columns[:-1]) + ["compass_x", "compass_y"])

    return data_df_s


def gaussian_filtering_causal(signal):
    gaussian_filter = gaussian(5, 5)
    causal_gaussian_filter = np.concatenate([np.zeros_like(gaussian_filter), gaussian_filter])
    filtered_signal = filters.convolve1d(signal, causal_gaussian_filter / causal_gaussian_filter.sum())
    return filtered_signal


def gaussian_filtering(signal):
    gaussian_filter = gaussian(10, 5)
    filtered_signal = filters.convolve1d(signal, gaussian_filter / gaussian_filter.sum())
    return filtered_signal
