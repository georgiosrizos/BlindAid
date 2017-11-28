########################################################################################################################
# Authors:  Dan Iorga, Georgios Rizos, Georgios Theodorakis, Johannes Wiebe, Thomas Uriot
#
# BlindAid: HiPEDS CDT group project - cohort 2017 - Imperial College London
########################################################################################################################

########################################################################################################################
# Dummy data generator of automated direction commands to micro:bit. Used for testing.
########################################################################################################################
import random

import serial
import signal

from python.blind_aid import utility


signal.signal(signal.SIGINT, utility.signal_handler)

PORT = "COM10"

BAUD = 115200

s = serial.Serial(PORT)
s.baudrate = BAUD
s.parity = serial.PARITY_NONE
s.databits = serial.EIGHTBITS
s.stopbits = serial.STOPBITS_ONE

try:
    s.reset_input_buffer()
    print("Start sending dummy data: \n")
    while True:
        # assume 0 == left, 1 == straight, 2 == right
        #  Wait for 3 seconds
        irand = random.randint(0, 2)
        print('The generator sent: ', irand)
        s.write(str(irand) + '\n')

finally:
    s.close()
