########################################################################################################################
# Authors:  Dan Iorga, Georgios Rizos, Georgios Theodorakis, Johannes Wiebe, Thomas Uriot
#
# BlindAid: HiPEDS CDT group project - cohort 2017 - Imperial College London
########################################################################################################################

########################################################################################################################
# Generates poster figure depicting trajectory-based approach results.
########################################################################################################################

import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from python.blind_aid import data_cleaning, utility

sns.set_style("darkgrid")
sns.set_context("paper")


def train_point_based_regression_model():
    list_of_data = list()
    list_of_labels = list()
    label = 0

    column_names = utility.get_column_names()
    features_names = utility.get_all_features_names()

    for i, filepath in enumerate(sorted(ml_approach_filepaths)):
        data_df, data = data_cleaning.read_csv_and_filter_missing(filepath, column_names)
        data_df = data_df[features_names]
        data = data_df.values

        list_of_data.append(data)
        list_of_labels.append(np.ones(shape=(data.shape[0], 1), dtype=np.int32) * label)

        label += 1

    X = np.vstack(list_of_data)
    y = np.vstack(list_of_labels)

    np.random.seed(42)
    perm = np.random.permutation(np.arange(X.shape[0]))
    X = X[perm, :]
    y = y[perm]

    classes_to_coordinates = dict()
    classes_to_coordinates[0] = [0, 0]
    classes_to_coordinates[1] = [0, 1]
    classes_to_coordinates[2] = [0, 2]
    classes_to_coordinates[3] = [0, 3]
    classes_to_coordinates[4] = [1, 0]
    classes_to_coordinates[5] = [1, 1]
    classes_to_coordinates[6] = [1, 2]
    classes_to_coordinates[7] = [1, 3]
    classes_to_coordinates[8] = [2, 0]
    classes_to_coordinates[9] = [2, 1]
    classes_to_coordinates[10] = [2, 2]
    classes_to_coordinates[11] = [2, 3]
    classes_to_coordinates[12] = [3, 0]
    classes_to_coordinates[13] = [3, 1]
    classes_to_coordinates[14] = [3, 2]
    classes_to_coordinates[15] = [3, 3]

    y_cl = y
    y = np.empty((X.shape[0], 2), dtype=np.int32)
    for i, yy in enumerate(list(y_cl[:, 0])):
        coordinates = classes_to_coordinates[yy]
        y[i, 0] = coordinates[0]
        y[i, 1] = coordinates[1]

    point_based_model = RandomForestRegressor(n_estimators=10)
    point_based_model.fit(X, y)
    return point_based_model


if __name__ == "__main__":
    ####################################################################################################################
    # Set dataset directories.
    ####################################################################################################################
    data_folder = utility.get_package_path() + "/datasets"

    filenames = os.listdir(data_folder)

    ml_approach_filepaths = [data_folder + "/" + filename for filename in filenames if "position" in filename]
    trajectory_approach_filepaths = [data_folder + "/" + filename for filename in filenames if "trajectory" in filename]

    ####################################################################################################################
    # Train point-based regression model.
    ####################################################################################################################
    point_based_model = train_point_based_regression_model()

    ####################################################################################################################
    # Read trajectory data.
    ####################################################################################################################
    features_names = utility.get_all_features_names()
    print(len(features_names))
    list_of_data = list()
    list_of_labels = list()
    for i, filepath in enumerate(trajectory_approach_filepaths):
        data_df, data = data_cleaning.read_csv_and_filter_missing(filepath, utility.get_column_names())

        data = data_df.values

        # Construct the compass labels.
        compass_labels = np.empty((data.shape[0], 2), dtype=np.float32)
        compass_labels[:, 0] = np.flip(data_cleaning.gaussian_filtering_causal(np.flip(data_df["compass_x"].values, axis=0)), axis=0)
        compass_labels[:, 1] = np.flip(data_cleaning.gaussian_filtering_causal(np.flip(data_df["compass_y"].values, axis=0)), axis=0)
        list_of_labels.append(compass_labels)

        # Augment features using point-based model.
        data_df = data_df[features_names]
        data = data_df.values

        # Filter data.
        for j in range(data.shape[1]):
            data[:, j] = data_cleaning.gaussian_filtering(data[:, j])

        position_pred = point_based_model.predict(data)
        augmented_data = np.hstack([data, position_pred])

        list_of_data.append(augmented_data)

    list_of_data = np.array(list_of_data)
    list_of_labels = np.array(list_of_labels)

    np.random.seed(42)
    perm = np.random.permutation(np.arange(len(list_of_data)))
    list_of_data = list_of_data[perm]
    list_of_labels = list_of_labels[perm]

    cv = KFold(n_splits=len(list_of_data), random_state=42)
    result_list_mse_mean = list()
    result_list_mse_std = list()

    X_dummy = np.empty((len(list_of_data), 15))
    y_dummy = np.empty((len(list_of_data), 1))

    results_list = list()

    f, axes = plt.subplots(3, 1, figsize=(14, 9))
    palette_seed = np.linspace(0, 3, 10)
    palette_counter = 0
    plot_counter = 0

    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=20)

    for train, test in cv.split(X_dummy, y_dummy):
        model = RandomForestRegressor(n_estimators=10)

        X_train = np.vstack(list_of_data[train])
        y_train = np.concatenate(list_of_labels[train])
        X_test = np.vstack(list_of_data[test])
        y_test = np.concatenate(list_of_labels[test])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        for i in range(y_pred.shape[0]):
            y_pred[i, :] = y_pred[i, :]/np.linalg.norm(y_pred[i, :])

        angle_test = list()
        angle_pred = list()

        for i in range(y_test.shape[0]):
            angle_test.append(np.degrees(np.arctan2(y_test[i, 0], y_test[i, 1])))
            angle_pred.append(np.degrees(np.arctan2(y_pred[i, 0], y_pred[i, 1])))

        angle_test = np.array(angle_test) + 180
        angle_pred = np.array(angle_pred) + 180

        # f.suptitle("Future Direction Prediction for Square Trajectory", fontsize=16)
        trials_to_visualize = [0, 4, 6]

        if plot_counter == trials_to_visualize[0]:
            axes[0].plot(np.arange(y_test.shape[0]), angle_test, "b", label="true")
            axes[0].plot(np.arange(y_pred.shape[0]), angle_pred, "r", label="pred")
            axes[0].set_title("trial #1", size=20)
            axes[0].set_ylabel("degrees", size=20)
            axes[0].set(ylim=(-10, 370))
            axes[0].legend(fontsize=20, loc="lower right")
            axes[0].tick_params(labelsize=15)

        if plot_counter == trials_to_visualize[1]:
            axes[1].plot(np.arange(y_test.shape[0]), angle_test, "b")
            axes[1].plot(np.arange(y_pred.shape[0]), angle_pred, "r")
            axes[1].set_title("trial #5", size=20)
            axes[1].set_ylabel("degrees", size=20)
            axes[1].set(ylim=(-10, 370))
            axes[1].tick_params(labelsize=15)

        if plot_counter == trials_to_visualize[2]:
            axes[2].plot(np.arange(y_test.shape[0]), angle_test, "b")
            axes[2].plot(np.arange(y_pred.shape[0]), angle_pred, "r")
            axes[2].set_title("trial #7", fontsize=20)
            axes[2].set_xlabel("time step (#)", size=20)
            axes[2].set_ylabel("degrees", size=20)
            axes[2].set(ylim=(-10, 370))
            axes[2].tick_params(labelsize=15)
            plt.tight_layout()
            plt.show()

        plot_counter += 1

        mse = np.mean(np.sqrt(np.abs(y_pred - y_test)))
