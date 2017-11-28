########################################################################################################################
# Authors:  Dan Iorga, Georgios Rizos, Georgios Theodorakis, Johannes Wiebe, Thomas Uriot
#
# BlindAid: HiPEDS CDT group project - cohort 2017 - Imperial College London
########################################################################################################################

########################################################################################################################
# Computer based data analysis. Receives measurement data, calculates desired direction and sends it to micro:bit.
########################################################################################################################

import os
import time

import serial
import signal
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from python.blind_aid import data_cleaning, utility

old_sent_timestamp = time.time()


if __name__ == "__main__":
    ####################################################################################################################
    # Setup serial library parameters.
    ####################################################################################################################
    signal.signal(signal.SIGINT, utility.signal_handler)

    PORT = "COM10"

    BAUD = 115200

    s = serial.Serial(PORT)
    s.baudrate = BAUD
    s.parity = serial.PARITY_NONE
    s.databits = serial.EIGHTBITS
    s.stopbits = serial.STOPBITS_ONE

    ####################################################################################################################
    # Define data filepaths.
    ####################################################################################################################
    data_folder = utility.get_package_path() + "/datasets"

    filenames = os.listdir(data_folder)

    ml_approach_filepaths = [data_folder + "/" + filename for filename in filenames if "position" in filename]
    trajectory_approach_filepaths = [data_folder + "/" + filename for filename in filenames if "trajectory" in filename]

    column_names = utility.get_column_names()

    ####################################################################################################################
    # Train point-based machine learning model.
    ####################################################################################################################
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

    print("Point-based ML model trained.")

    ####################################################################################################################
    # Train trajectory-based machine learning model.
    ####################################################################################################################
    features_names = utility.get_all_features_names()
    print(len(features_names))
    list_of_data = list()
    list_of_labels = list()
    for i, filepath in enumerate(trajectory_approach_filepaths):
        data_df, data = data_cleaning.read_csv_and_filter_missing(filepath, column_names)

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

        label += 1

    X = np.vstack(list_of_data)
    y = np.concatenate(list_of_labels)

    trajectory_based_model = RandomForestRegressor(n_estimators=10)
    trajectory_based_model.fit(X, y)

    print("Trajectory-based ML model trained.")

    ####################################################################################################################
    # Initialize data buffer.
    ####################################################################################################################
    meas_list = list()
    curr_meas_chunk = np.ones((1, 17), dtype=np.int32) * -9999
    meas_list.append(curr_meas_chunk)

    timestamp_to_id = dict()
    id_to_timestamp = dict()
    checkpoint = 0

    all_features_column_index = np.array([1,
                                          2,
                                          3,
                                          4,
                                          5,
                                          6,
                                          7,
                                          8,
                                          12,
                                          13,
                                          14,
                                          15], dtype=np.int32)

    features_names = utility.get_all_features_names()

    direction_list = list()
    direction_counter = 0
    made_prediction_for = -1

    trial_name = "demo_1"
    measurement_filepath = trial_name + "_measurement.csv"
    direction_filepath = trial_name + "_direction.csv"

    try:
        s.reset_input_buffer()

        # distance = raw_input("Enter distance:")

        print("Start sending data: \n")
        while True:
            ############################################################################################################
            # Read microbit data and put them in buffer.
            ############################################################################################################
            # read a line from the microbit, decode it and
            # strip the whitespace at the end
            data = s.readline().rstrip()
            data = data.decode("ascii")

            # split the data
            data_s = data.split("_")
            if len(data_s) == 2:
                timestamp = int(data_s[0])

                if int(data_s[0]) == -1:
                    print("CHECKPOINT: " + data)
                    checkpoint = int(val)
                    continue

                msg_id = timestamp_to_id.get(timestamp, len(timestamp_to_id))
                timestamp_to_id[timestamp] = msg_id
                id_to_timestamp[msg_id] = timestamp
                data_ss = data_s[1].split(":")
                val_id = int(data_ss[0])
                val = data_ss[1]

                # if val_id == val_to_print:
                #     print(val)

                if msg_id >= len(meas_list):
                    offset = msg_id - len(meas_list) + 1
                    for oo, oo_msg_id in zip(range(offset), range(msg_id, msg_id + offset)):
                        meas_list.append(np.ones((1, 17), dtype=np.int32) * -9999)
                        meas_list[-1][0, 0] = checkpoint
                        meas_list[-1][0, -1] = id_to_timestamp[oo_msg_id]
                        direction_counter += 1
                meas_list[msg_id][0, val_id] = int(val)

            if direction_counter > made_prediction_for:
                ########################################################################################################
                # Clean and filter data.
                ########################################################################################################
                # Get the most recent data.
                X = np.vstack(meas_list[-15:])
                X_df, X = data_cleaning.filter_missing(X, column_names)
                X_df = X_df[features_names]
                X = X_df.values

                # Filter data.
                for j in range(X.shape[1]):
                    X[:, j] = data_cleaning.gaussian_filtering(X[:, j])

                # Augment data with the point-based model.
                position_pred = point_based_model.predict(X[-2:, :])
                augmented_X = np.hstack([X[-2:, :], position_pred])
                augmented_X = augmented_X[-1, :]
                augmented_X = augmented_X / np.linalg.norm(augmented_X)

                ########################################################################################################
                # Make direction prediction and translate to user-understandable cue.
                ########################################################################################################
                made_prediction_for = direction_counter
                y_pred = trajectory_based_model.predict(augmented_X)
                y_pred = y_pred / np.linalg.norm(y_pred)

                current_angle = np.degrees(np.arctan2(augmented_X[-2], augmented_X[-1]))
                predicted_angle = np.degrees(np.arctan2(y_pred[0, 0], y_pred[0, 1]))

                angle_change = current_angle - predicted_angle

                if angle_change > 180:
                    angle_change = angle_change - 360
                elif angle_change < -180:
                    angle_change = angle_change + 360

                # assume 0 == left, 1 == straight, 2 == right
                if angle_change < -10:
                    i_direction = 2
                elif angle_change < 10:
                    i_direction = 1
                elif angle_change < 180:
                    i_direction = 0
                else:
                    i_direction = 1
                    print("Invalid angle value.")
                direction_list.append(i_direction)

            # Wait for 3 seconds
            # time.sleep(3)

            sent_timestamp = time.time()
            # print(sent_timestamp - old_sent_timestamp)
            if sent_timestamp - old_sent_timestamp > 3:
                print("The generator sent: ", i_direction, angle_change)
                s.write(str(i_direction) + '\n')
                old_sent_timestamp = sent_timestamp

    finally:
        s.close()
        meas_list = np.vstack(meas_list)
        with open(measurement_filepath, "wb") as fp:
            np.savetxt(fp, meas_list, fmt='%i', delimiter=",")
        #
        # with open(direction_filepath, "w") as fp:
        #     for i_direction in direction_list:
        #         fp.write(repr(i_direction) + "\n")


