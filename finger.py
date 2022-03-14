"""
See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.argrelmin.html
"""
import numpy as np
from scipy.signal import argrelmin

def detect(sensor_data: np.array):
    min_idxs = argrelmin(sensor_data)
    values = []
    for mini in min_idxs[0]:
        values.append((sensor_data[mini], mini))
    return values

