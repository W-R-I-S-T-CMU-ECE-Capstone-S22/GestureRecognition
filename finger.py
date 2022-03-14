"""
See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.argrelmin.html
"""
import numpy as np
from scipy.signal import argrelmin


def detect(sensor_data):
    mod_sensors = []
    for sensor in sensor_data:
        if(sensor < 200):
            mod_sensors.append(sensor)

    # bad way to append 255 to the first and last values of the array
    mod_sensors = np.array(mod_sensors)
    mod_sensors = np.append(mod_sensors, 255)
    mod_sensors = np.append(255, mod_sensors)

    min_idxs = argrelmin(np.array(mod_sensors))
    values = []
    for mini in min_idxs[0]:
        val = np.where(sensor_data == mod_sensors[mini])[0]
        values.append((sensor_data[val], val))
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
