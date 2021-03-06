"""
Wrapper for sensor data to be read from a file.
"""

import time
import numpy as np
from constants import *


class BatteryInfo:
    def __init__(self, data):
        timestamp, voltage, percentage = data.decode("utf-8").split(",")
        self.timestamp = int(timestamp)
        self.voltage = float(voltage)
        self.percentage = float(percentage)


class SensorData:
    last_timestamp = None

    def __init__(self, data):
        data = list(data)
        if len(data) > 0:
            if SensorData.last_timestamp is None:
                SensorData.last_timestamp = time.time()

            self.timestamp = SensorData.last_timestamp + data[0] / 1000
            SensorData.last_timestamp = self.timestamp
            self.raw = np.array(data[-NUM_SENSORS:])
            self.raw[self.raw > DIST_THRES] = 255
        else:
            self.timestamp = SensorData.last_timestamp
            self.raw = 255 * np.ones(NUM_SENSORS)

    @classmethod
    def get_sensors(cls):
        return np.arange(NUM_SENSORS) * SENSOR_DIST


class SensorDatasFromFile:
    def __init__(self, filename):
        self.timestamps = []
        self.raw = []

        self.read_file(filename)

    def read_file(self, filename):
        f = open(filename, "r")
        for data in f.read().split("\n"):
            if data == "":
                continue

            timestamp_idx = data.index(", [")
            timestamp = float(data[:timestamp_idx])
            self.timestamps += [timestamp]

            raw_str = data[timestamp_idx:].strip(", ").strip("[").strip("]")
            raw = np.fromstring(raw_str, dtype=np.uint8, sep=", ")
            raw[raw > DIST_THRES] = 255
            self.raw += [raw]

