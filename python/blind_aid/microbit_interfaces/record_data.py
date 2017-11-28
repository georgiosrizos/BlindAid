########################################################################################################################
# Authors:  Dan Iorga, Georgios Rizos, Georgios Theodorakis, Johannes Wiebe, Thomas Uriot
#
# BlindAid: HiPEDS CDT group project - cohort 2017 - Imperial College London
########################################################################################################################

########################################################################################################################
# We used this script for recording measurements during our training trials.
########################################################################################################################

import serial

import signal
import numpy as np

from python.blind_aid import utility


signal.signal(signal.SIGINT, utility.signal_handler)


PORT = "COM10"

BAUD = 115200

s = serial.Serial(PORT)
s.baudrate = BAUD
s.parity = serial.PARITY_NONE
s.databits = serial.EIGHTBITS
s.stopbits = serial.STOPBITS_ONE

file_string = "data_test22.csv"

Nm = 250
i = 0

meas_list = list()
curr_meas_chunk = np.ones((1, 17), dtype=np.int32) * -9999
meas_list.append(curr_meas_chunk)
checkpoint = 0
timestamp_to_id = dict()
id_to_timestamp = dict()

val_to_print = 6
lmin = 0
lmax = 3600


try:
    s.reset_input_buffer()

    while True:
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

            if val_id == val_to_print:
                print(val)

            if msg_id >= len(meas_list):
                offset = msg_id - len(meas_list) + 1
                for oo, oo_msg_id in zip(range(offset), range(msg_id, msg_id + offset)):
                    meas_list.append(np.ones((1, 17), dtype=np.int32) * -9999)
                    meas_list[-1][0, 0] = checkpoint
                    meas_list[-1][0, -1] = id_to_timestamp[oo_msg_id]
            meas_list[msg_id][0, val_id] = int(val)


finally:
    s.close()
    meas_list = np.vstack(meas_list)
    # print(meas_list)
    with open(file_string, "wb") as fp:
        np.savetxt(fp, meas_list, fmt='%i', delimiter=",")

