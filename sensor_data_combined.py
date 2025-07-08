from mbientlab.metawear import MetaWear, libmetawear
from mbientlab.metawear.cbindings import *
from mbientlab.warble import *
from ctypes import byref, POINTER, cast
from threading import Event
import csv
from time import time, sleep


# Settings
MAC_ADDRESS = "ED:72:F8:F9:37:40"
CSV_FILENAME = "C:/Users/Tzotzo Pc/IoT_PROJECT/data/classB_wave/classB_.csv"
DURATION_SECONDS = 10


# Logging and state
data_log = []
latest = {"acc": [None, None, None], "gyro": [None, None, None]}
e = Event()


# Create CSV file with headers
with open(CSV_FILENAME, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"])


# Save start time for timestamp
start_time = time()


# Callback for accelerometer
def acc_callback(ctx, data):
    acc = cast(data.contents.value, POINTER(CartesianFloat)).contents
    latest["acc"] = [round(acc.x, 5), round(acc.y, 5), round(acc.z, 5)]
    combine_and_store()


# Callback for gyroscope
def gyro_callback(ctx, data):
    gyro = cast(data.contents.value, POINTER(CartesianFloat)).contents
    latest["gyro"] = [round(gyro.x, 5), round(gyro.y, 5), round(gyro.z, 5)]
    combine_and_store()


# Combine acc + gyro measurements
def combine_and_store():
    if None not in latest["acc"] + latest["gyro"]:
        timestamp = round(time() - start_time, 5)
        row = [timestamp] + latest["acc"] + latest["gyro"]
        data_log.append(row)
        # print(f"Row: {row}")  # Enable if needed


# -------- Connection --------
print("Connecting to", MAC_ADDRESS, "...")
device = MetaWear(MAC_ADDRESS)
device.connect()
print("Connected!")


# -------- Accelerometer --------
libmetawear.mbl_mw_acc_set_odr(device.board, 100.0)
libmetawear.mbl_mw_acc_set_range(device.board, 4.0)
libmetawear.mbl_mw_acc_write_acceleration_config(device.board)


acc_signal = libmetawear.mbl_mw_acc_get_acceleration_data_signal(device.board)
acc_cb = FnVoid_VoidP_DataP(acc_callback)
libmetawear.mbl_mw_datasignal_subscribe(acc_signal, None, acc_cb)


libmetawear.mbl_mw_acc_enable_acceleration_sampling(device.board)
libmetawear.mbl_mw_acc_start(device.board)


# -------- Gyroscope --------
libmetawear.mbl_mw_gyro_bmi160_set_odr(device.board, 7)   # 100Hz
libmetawear.mbl_mw_gyro_bmi160_set_range(device.board, 1) # Â±500 dps
libmetawear.mbl_mw_gyro_bmi160_write_config(device.board)


gyro_signal = libmetawear.mbl_mw_gyro_bmi160_get_rotation_data_signal(device.board)
gyro_cb = FnVoid_VoidP_DataP(gyro_callback)
libmetawear.mbl_mw_datasignal_subscribe(gyro_signal, None, gyro_cb)


libmetawear.mbl_mw_gyro_bmi160_enable_rotation_sampling(device.board)
libmetawear.mbl_mw_gyro_bmi160_start(device.board)


# -------- Stream --------
print(f"Streaming data for {DURATION_SECONDS} seconds...")
sleep(DURATION_SECONDS)


# -------- Saving --------
print("Saving to", CSV_FILENAME)
with open(CSV_FILENAME, mode='a', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data_log)


# -------- Termination --------
print("Stopping...")
libmetawear.mbl_mw_acc_stop(device.board)
libmetawear.mbl_mw_acc_disable_acceleration_sampling(device.board)
libmetawear.mbl_mw_datasignal_unsubscribe(acc_signal)


libmetawear.mbl_mw_gyro_bmi160_stop(device.board)
libmetawear.mbl_mw_gyro_bmi160_disable_rotation_sampling(device.board)
libmetawear.mbl_mw_datasignal_unsubscribe(gyro_signal)


device.disconnect()
print("Disconnected.")
print("Done.")