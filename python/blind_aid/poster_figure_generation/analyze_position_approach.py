########################################################################################################################
# Authors:  Dan Iorga, Georgios Rizos, Georgios Theodorakis, Johannes Wiebe, Thomas Uriot
#
# BlindAid: HiPEDS CDT group project - cohort 2017 - Imperial College London
########################################################################################################################

########################################################################################################################
# Generates poster figure depicting position-based approach results.
########################################################################################################################

import os
import statistics
import time
import collections

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from scipy.spatial.distance import euclidean
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from python.blind_aid import data_cleaning, utility

sns.set_style("darkgrid")
sns.set_context("paper")


if __name__ == "__main__":
    data_folder = utility.get_package_path() + "/datasets"

    filenames = os.listdir(data_folder)

    ml_approach_filepaths = [data_folder + "/" + filename for filename in filenames if "position" in filename]
    trajectory_approach_filepaths = [data_folder + "/" + filename for filename in filenames if "trajectory" in filename]

    column_names = utility.get_column_names()

    list_of_data = list()
    list_of_labels = list()
    label = 0

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
    sample_index = np.arange(X.shape[0])
    np.random.seed(42)
    perm = np.random.permutation(np.arange(X.shape[0]))
    X = X[perm, :]
    y = y[perm]
    sample_index = sample_index[perm]

    ####################################################################################################################
    # Machine learning approach - regression.
    ####################################################################################################################
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

    coordinates_to_classes = dict()
    coordinates_to_classes["0_0"] = 0
    coordinates_to_classes["0_1"] = 1
    coordinates_to_classes["0_2"] = 2
    coordinates_to_classes["0_3"] = 3
    coordinates_to_classes["1_0"] = 4
    coordinates_to_classes["1_1"] = 5
    coordinates_to_classes["1_2"] = 6
    coordinates_to_classes["1_3"] = 7
    coordinates_to_classes["2_0"] = 8
    coordinates_to_classes["2_1"] = 9
    coordinates_to_classes["2_2"] = 10
    coordinates_to_classes["2_3"] = 11
    coordinates_to_classes["3_0"] = 12
    coordinates_to_classes["3_1"] = 13
    coordinates_to_classes["3_2"] = 14
    coordinates_to_classes["3_3"] = 15

    y_cl = y
    y = np.empty((X.shape[0], 2), dtype=np.int32)
    for i, yy in enumerate(list(y_cl[:, 0])):
        coordinates = classes_to_coordinates[yy]
        y[i, 0] = coordinates[0]
        y[i, 1] = coordinates[1]

    cv = KFold(n_splits=10, random_state=42)
    result_list_mse_mean = list()
    result_list_mse_std = list()

    coordinates_to_results = collections.defaultdict(list)
    coordinates_to_points = collections.defaultdict(list)

    for train, test in cv.split(X, y):

        start_time = time.perf_counter()
        model = RandomForestRegressor(n_estimators=10)
        model.fit(X[train], y[train])
        y_pred = model.predict(X[test])
        elapsed_time = time.perf_counter() - start_time
        # print(elapsed_time)

        within_fold_distances = list()
        for i, t, p in zip(list(test), list(y[test]), list(y_pred)):
            within_fold_distances.append(euclidean(t, p))

            coordinates_str = repr(t[0]) + "_" + repr(t[1])
            coordinates_to_points[coordinates_str].append(np.array(p))
            coordinates_to_results[coordinates_str].append(within_fold_distances[-1])

        result_list_mse_mean.append(statistics.mean(within_fold_distances))
        result_list_mse_std.append(statistics.stdev(within_fold_distances))

    for k in coordinates_to_points.keys():
        coordinates_to_points[k] = np.vstack(coordinates_to_points[k])
        coordinates_to_points[k] = coordinates_to_points[k] + (np.random.randn(coordinates_to_points[k].shape[0], coordinates_to_points[k].shape[1]) * 0.05)

    all_features_mean = statistics.mean(result_list_mse_mean)
    all_features_std = statistics.mean(result_list_mse_std)
    print("Distance:", statistics.mean(result_list_mse_mean), "+-", statistics.mean(result_list_mse_std))

    f, axes = plt.subplots(3, 3, figsize=(9, 9))
    palette_seed = np.linspace(0, 3, 10)
    palette_counter = 0

    k = "0_0"
    cmap = sns.cubehelix_palette(start=palette_counter, light=1, as_cmap=True)
    axes[0, 2].add_patch(patches.Rectangle((0, 0),
                                           3,
                                           3,
                                           fill=False,
                                           linewidth=1))
    axes[0, 2].scatter(classes_to_coordinates[0][0],
                       classes_to_coordinates[0][1], marker="h", color="k")
    axes[0, 2].scatter(coordinates_to_points[k][:, 0], coordinates_to_points[k][:, 1], marker=".", color="r")
    # sns.kdeplot(coordinates_to_points[k][:, 0],
    #             coordinates_to_points[k][:, 1], cmap=cmap, shade=False, ax=axes[0, 2])
    axes[0, 2].set(xlim=(-3, 3), ylim=(-3, 3))
    axes[0, 2].set_title("R+C+S", fontsize=20)
    axes[0, 2].annotate("MDE=" + str(statistics.mean(coordinates_to_results[k]))[:4],
                        xy=(0, -2), xycoords='data',
                        xytext=(0, -2), textcoords='data',
                        size=20, va="center", ha="center",
                        bbox=dict(boxstyle="round4", fc="w"),
                        arrowprops=dict(arrowstyle="-|>",
                                        connectionstyle="arc3,rad=-0.2",
                                        fc="w"),
                        )
    axes[0, 2].tick_params(labelsize=9)
    palette_counter += 1

    k = "2_0"
    cmap = sns.cubehelix_palette(start=palette_counter, light=1, as_cmap=True)
    axes[1, 2].add_patch(patches.Rectangle((0, 0),
                                           3,
                                           3,
                                           fill=False,
                                           linewidth=1))
    axes[1, 2].scatter(classes_to_coordinates[8][0],
                       classes_to_coordinates[8][1], marker="h", color="k")
    axes[1, 2].scatter(coordinates_to_points[k][:, 0], coordinates_to_points[k][:, 1], marker=".", color="g")
    # sns.kdeplot(coordinates_to_points[k][:, 0],
    #             coordinates_to_points[k][:, 1], cmap=cmap, shade=False, ax=axes[1, 2])
    axes[1, 2].set(xlim=(-1, 5), ylim=(-3, 3))
    axes[1, 2].annotate("MDE=" + str(statistics.mean(coordinates_to_results[k]))[:4],
                        xy=(2, -2), xycoords='data',
                        xytext=(2, -2), textcoords='data',
                        size=20, va="center", ha="center",
                        bbox=dict(boxstyle="round4", fc="w"),
                        arrowprops=dict(arrowstyle="-|>",
                                        connectionstyle="arc3,rad=-0.2",
                                        fc="w"),
                        )
    axes[1, 2].tick_params(labelsize=9)
    palette_counter += 1

    k = "1_1"
    cmap = sns.cubehelix_palette(start=palette_counter, light=1, as_cmap=True)
    axes[2, 2].add_patch(patches.Rectangle((0, 0),
                                           3,
                                           3,
                                           fill=False,
                                           linewidth=1))
    axes[2, 2].scatter(classes_to_coordinates[5][0],
                       classes_to_coordinates[5][1], marker="h", color="k")
    axes[2, 2].scatter(coordinates_to_points[k][:, 0], coordinates_to_points[k][:, 1], marker=".", color="b")
    # sns.kdeplot(coordinates_to_points[k][:, 0],
    #             coordinates_to_points[k][:, 1], cmap=cmap, shade=False, ax=axes[2, 2])
    axes[2, 2].set(xlim=(-2, 4), ylim=(-2, 4))
    axes[2, 2].set_xlabel("metres", size=20)
    axes[2, 2].annotate("MDE=" + str(statistics.mean(coordinates_to_results[k]))[:4],
                        xy=(1, -1), xycoords='data',
                        xytext=(1, -1), textcoords='data',
                        size=20, va="center", ha="center",
                        bbox=dict(boxstyle="round4", fc="w"),
                        arrowprops=dict(arrowstyle="-|>",
                                        connectionstyle="arc3,rad=-0.2",
                                        fc="w"),
                        )
    axes[2, 2].tick_params(labelsize=9)
    palette_counter += 1



    list_of_data = list()
    list_of_labels = list()
    label = 0

    features_names = utility.get_radio_features_names()

    for i, filepath in enumerate(sorted(ml_approach_filepaths)):
        data_df, data = data_cleaning.read_csv_and_filter_missing(filepath, column_names)
        data_df = data_df[features_names]
        # data_df["sonar_1"].plot()
        # plt.show()
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

    y_cl = y
    y = np.empty((X.shape[0], 2), dtype=np.int32)
    for i, yy in enumerate(list(y_cl[:, 0])):
        coordinates = classes_to_coordinates[yy]
        y[i, 0] = coordinates[0]
        y[i, 1] = coordinates[1]

    # print(y)

    cv = KFold(n_splits=10, random_state=42)
    result_list_mse_mean = list()
    result_list_mse_std = list()

    coordinates_to_results = collections.defaultdict(list)
    coordinates_to_points = collections.defaultdict(list)

    for train, test in cv.split(X, y):

        start_time = time.perf_counter()
        model = RandomForestRegressor(n_estimators=10)
        model.fit(X[train], y[train])
        y_pred = model.predict(X[test])
        elapsed_time = time.perf_counter() - start_time
        # print(elapsed_time)

        within_fold_distances = list()
        for t, p in zip(list(y[test]), list(y_pred)):
            within_fold_distances.append(euclidean(t, p))

            coordinates_str = repr(t[0]) + "_" + repr(t[1])
            coordinates_to_points[coordinates_str].append(np.array(p))
            coordinates_to_results[coordinates_str].append(within_fold_distances[-1])

        result_list_mse_mean.append(statistics.mean(within_fold_distances))
        result_list_mse_std.append(statistics.stdev(within_fold_distances))

    for k in coordinates_to_points.keys():
        coordinates_to_points[k] = np.vstack(coordinates_to_points[k])
        coordinates_to_points[k] = coordinates_to_points[k] + (
        np.random.randn(coordinates_to_points[k].shape[0], coordinates_to_points[k].shape[1]) * 0.05)

    radio_features_mean = statistics.mean(result_list_mse_mean)
    radio_features_std = statistics.mean(result_list_mse_std)
    print("Distance:", statistics.mean(result_list_mse_mean), "+-", statistics.mean(result_list_mse_std))

    palette_counter = 0

    k = "0_0"
    cmap = sns.cubehelix_palette(start=palette_counter, light=1, as_cmap=True)
    axes[0, 0].add_patch(patches.Rectangle((0, 0),
                                           3,
                                           3,
                                           fill=False,
                                           linewidth=1))
    axes[0, 0].scatter(classes_to_coordinates[0][0],
                       classes_to_coordinates[0][1], marker="h", color="k")
    axes[0, 0].scatter(coordinates_to_points[k][:, 0], coordinates_to_points[k][:, 1], marker=".", color="r")
    # sns.kdeplot(coordinates_to_points[k][:, 0],
    #             coordinates_to_points[k][:, 1], cmap=cmap, shade=False, ax=axes[0, 0])
    axes[0, 0].set(xlim=(-3, 3), ylim=(-3, 3))
    axes[0, 0].set_title("R", fontsize=20)
    axes[0, 0].set_ylabel("metres", size=20)
    axes[0, 0].annotate("MDE=" + str(statistics.mean(coordinates_to_results[k]))[:4],
                        xy=(0, -2), xycoords='data',
                        xytext=(0, -2), textcoords='data',
                        size=20, va="center", ha="center",
                        bbox=dict(boxstyle="round4", fc="w"),
                        arrowprops=dict(arrowstyle="-|>",
                                        connectionstyle="arc3,rad=-0.2",
                                        fc="w"),
                        )
    axes[0, 0].tick_params(labelsize=9)
    palette_counter += 1

    k = "2_0"
    cmap = sns.cubehelix_palette(start=palette_counter, light=1, as_cmap=True)
    axes[1, 0].add_patch(patches.Rectangle((0, 0),
                                           3,
                                           3,
                                           fill=False,
                                           linewidth=1))
    axes[1, 0].scatter(classes_to_coordinates[8][0],
                       classes_to_coordinates[8][1], marker="h", color="k")
    axes[1, 0].scatter(coordinates_to_points[k][:, 0], coordinates_to_points[k][:, 1], marker=".", color="g")
    # sns.kdeplot(coordinates_to_points[k][:, 0],
    #             coordinates_to_points[k][:, 1], cmap=cmap, shade=False, ax=axes[1, 0])
    axes[1, 0].set(xlim=(-1, 5), ylim=(-3, 3))
    axes[1, 0].set_ylabel("metres", size=20)
    axes[1, 0].annotate("MDE=" + str(statistics.mean(coordinates_to_results[k]))[:4],
                        xy=(2, -2), xycoords='data',
                        xytext=(2, -2), textcoords='data',
                        size=20, va="center", ha="center",
                        bbox=dict(boxstyle="round4", fc="w"),
                        arrowprops=dict(arrowstyle="-|>",
                                        connectionstyle="arc3,rad=-0.2",
                                        fc="w"),
                        )
    axes[1, 0].tick_params(labelsize=9)
    palette_counter += 1

    k = "1_1"
    cmap = sns.cubehelix_palette(start=palette_counter, light=1, as_cmap=True)
    axes[2, 0].add_patch(patches.Rectangle((0, 0),
                                           3,
                                           3,
                                           fill=False,
                                           linewidth=1))
    axes[2, 0].scatter(classes_to_coordinates[5][0],
                       classes_to_coordinates[5][1], marker="h", color="k")
    axes[2, 0].scatter(coordinates_to_points[k][:, 0], coordinates_to_points[k][:, 1], marker=".", color="b")
    # sns.kdeplot(coordinates_to_points[k][:, 0],
    #             coordinates_to_points[k][:, 1], cmap=cmap, shade=False, ax=axes[2, 0])
    axes[2, 0].set(xlim=(-2, 4), ylim=(-2, 4))
    axes[2, 0].set_ylabel("metres", size=20)
    axes[2, 0].set_xlabel("metres", size=20)
    axes[2, 0].annotate("MDE=" + str(statistics.mean(coordinates_to_results[k]))[:4],
                        xy=(1, -1), xycoords='data',
                        xytext=(1, -1), textcoords='data',
                        size=20, va="center", ha="center",
                        bbox=dict(boxstyle="round4", fc="w"),
                        arrowprops=dict(arrowstyle="-|>",
                                        connectionstyle="arc3,rad=-0.2",
                                        fc="w"),
                        )
    axes[2, 0].tick_params(labelsize=9)
    palette_counter += 1

    list_of_data = list()
    list_of_labels = list()
    label = 0

    features_names = utility.get_radio_compass_features_names()

    for i, filepath in enumerate(sorted(ml_approach_filepaths)):
        data_df, data = data_cleaning.read_csv_and_filter_missing(filepath, column_names)
        data_df = data_df[features_names]
        # data_df["sonar_1"].plot()
        # plt.show()
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

    y_cl = y
    y = np.empty((X.shape[0], 2), dtype=np.int32)
    for i, yy in enumerate(list(y_cl[:, 0])):
        coordinates = classes_to_coordinates[yy]
        y[i, 0] = coordinates[0]
        y[i, 1] = coordinates[1]

    # print(y)

    cv = KFold(n_splits=10, random_state=42)
    result_list_mse_mean = list()
    result_list_mse_std = list()

    coordinates_to_results = collections.defaultdict(list)
    coordinates_to_points = collections.defaultdict(list)

    for train, test in cv.split(X, y):

        start_time = time.perf_counter()
        model = RandomForestRegressor(n_estimators=10)
        model.fit(X[train], y[train])
        y_pred = model.predict(X[test])
        elapsed_time = time.perf_counter() - start_time
        # print(elapsed_time)

        within_fold_distances = list()
        for t, p in zip(list(y[test]), list(y_pred)):
            within_fold_distances.append(euclidean(t, p))

            coordinates_str = repr(t[0]) + "_" + repr(t[1])
            coordinates_to_points[coordinates_str].append(np.array(p))
            coordinates_to_results[coordinates_str].append(within_fold_distances[-1])

        result_list_mse_mean.append(statistics.mean(within_fold_distances))
        result_list_mse_std.append(statistics.stdev(within_fold_distances))

    for k in coordinates_to_points.keys():
        coordinates_to_points[k] = np.vstack(coordinates_to_points[k])
        coordinates_to_points[k] = coordinates_to_points[k] + (
            np.random.randn(coordinates_to_points[k].shape[0], coordinates_to_points[k].shape[1]) * 0.05)

    radio_compass_features_mean = statistics.mean(result_list_mse_mean)
    radio_compass_features_std = statistics.mean(result_list_mse_std)
    print("Distance:", statistics.mean(result_list_mse_mean), "+-", statistics.mean(result_list_mse_std))

    palette_counter = 0

    k = "0_0"
    cmap = sns.cubehelix_palette(start=palette_counter, light=1, as_cmap=True)
    axes[0, 1].add_patch(patches.Rectangle((0, 0),
                                           3,
                                           3,
                                           fill=False,
                                           linewidth=1))
    axes[0, 1].scatter(classes_to_coordinates[0][0],
                       classes_to_coordinates[0][1], marker="h", color="k")
    axes[0, 1].scatter(coordinates_to_points[k][:, 0], coordinates_to_points[k][:, 1], marker=".", color="r")
    # sns.kdeplot(coordinates_to_points[k][:, 0],
    #             coordinates_to_points[k][:, 1], cmap=cmap, shade=False, ax=axes[0, 1])


    # print(coordinates_to_points[k])
    # print(classes_to_coordinates[0])
    axes[0, 1].set(xlim=(-3, 3), ylim=(-3, 3))
    axes[0, 1].set_title("R+C", fontsize=20)
    axes[0, 1].annotate("MDE=" + str(statistics.mean(coordinates_to_results[k]))[:4],
                        xy=(0, -2), xycoords='data',
                        xytext=(0, -2), textcoords='data',
                        size=20, va="center", ha="center",
                        bbox=dict(boxstyle="round4", fc="w"),
                        arrowprops=dict(arrowstyle="-|>",
                                        connectionstyle="arc3,rad=-0.2",
                                        fc="w"),
                        )
    axes[0, 1].tick_params(labelsize=9)
    palette_counter += 1

    k = "2_0"
    cmap = sns.cubehelix_palette(start=palette_counter, light=1, as_cmap=True)
    axes[1, 1].add_patch(patches.Rectangle((0, 0),
                                           3,
                                           3,
                                           fill=False,
                                           linewidth=1))
    axes[1, 1].scatter(classes_to_coordinates[8][0],
                       classes_to_coordinates[8][1], marker="h", color="k")
    axes[1, 1].scatter(coordinates_to_points[k][:, 0], coordinates_to_points[k][:, 1], marker=".", color="g")
    # sns.kdeplot(coordinates_to_points[k][:, 0],
    #             coordinates_to_points[k][:, 1], cmap=cmap, shade=False, ax=axes[1, 1])
    axes[1, 1].set(xlim=(-1, 5), ylim=(-3, 3))
    axes[1, 1].annotate("MDE=" + str(statistics.mean(coordinates_to_results[k]))[:4],
                        xy=(2, -2), xycoords='data',
                        xytext=(2, -2), textcoords='data',
                        size=20, va="center", ha="center",
                        bbox=dict(boxstyle="round4", fc="w"),
                        arrowprops=dict(arrowstyle="-|>",
                                        connectionstyle="arc3,rad=-0.2",
                                        fc="w"),
                        )
    axes[1, 1].tick_params(labelsize=9)
    palette_counter += 1

    k = "1_1"
    cmap = sns.cubehelix_palette(start=palette_counter, light=1, as_cmap=True)
    axes[2, 1].add_patch(patches.Rectangle((0, 0),
                                           3,
                                           3,
                                           fill=False,
                                           linewidth=1))
    axes[2, 1].scatter(classes_to_coordinates[5][0],
                       classes_to_coordinates[5][1], marker="h", color="k")
    axes[2, 1].scatter(coordinates_to_points[k][:, 0], coordinates_to_points[k][:, 1], marker=".", color="b")
    # sns.kdeplot(coordinates_to_points[k][:, 0],
    #             coordinates_to_points[k][:, 1], cmap=cmap, shade=False, ax=axes[2, 1])
    axes[2, 1].set(xlim=(-2, 4), ylim=(-2, 4))
    axes[2, 1].set_xlabel("metres", size=20)
    axes[2, 1].annotate("MDE=" + str(statistics.mean(coordinates_to_results[k]))[:4],
                        xy=(1, -1), xycoords='data',
                        xytext=(1, -1), textcoords='data',
                        size=20, va="center", ha="center",
                        bbox=dict(boxstyle="round4", fc="w"),
                        arrowprops=dict(arrowstyle="-|>",
                                        connectionstyle="arc3,rad=-0.2",
                                        fc="w"),
                      )
    axes[2, 1].tick_params(labelsize=9)
    palette_counter += 1

    f.tight_layout()
    # f.suptitle("Modality Integration Impact", fontsize=20)
    plt.show()
