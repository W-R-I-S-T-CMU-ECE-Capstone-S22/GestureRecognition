"""
See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.argrelmin.html
"""
import numpy as np
from scipy.signal import argrelmin


def detect(sensor_data):
    #sensor_data = np.append(sensor_data, 255)
    #sensor_data = np.append(255, sensor_data)
    min_idxs = argrelmin(sensor_data)
    values = []
    for mini in min_idxs[0]:
        values.append((sensor_data[mini], mini))
    return values


# faulty finger detection algorithm
def faulty_detect(sensors):
    first_min = min(sensors)
    first_index = sensors.index(first_min)
    second_min = sorted(set(sensors))[1]
    second_index = sensors.index(second_min)
    minimums = []
    minimums.append((first_min, first_index))
    if (abs(first_index-second_index) > 2):
        minimums.append((second_min, second_index))
    return minimums
