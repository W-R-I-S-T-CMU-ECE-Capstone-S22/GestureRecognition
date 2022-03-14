"""
Wrapper for sensor data to be read from a file.
"""

import numpy as np

class SensorData:
    def __init__(self, filename):
        self.timestamps = []
        self.raw = []

        self.read_file(filename)

    def read_file(self, filename):
        f = open(filename, "r")
        for data in f.read().split("\n"):
            if data == "": continue

            timestamp_idx = data.index(", [")
            timestamp = float(data[:timestamp_idx])
            self.timestamps += [timestamp]

            raw_str = data[timestamp_idx:].strip(", ").strip("[").strip("]")
            raw = np.fromstring(raw_str, dtype=np.uint8, sep=", ")
            self.raw += [raw]

