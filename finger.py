"""
"""
import numpy as np
import scipy.signal
import scipy.optimize
import matplotlib.pyplot as plt

from constants import *
from sensor_data import SensorData

EXTRA_LEN = 2


def remove_bad_data(sensor_data):
    idxs = np.where(sensor_data >= DIST_THRES)
    new_data = np.delete(sensor_data, idxs)
    return new_data, idxs


def find_finger_y(finger_idx, sensor_data):
    finger_y = finger_idx

    center_val = float(sensor_data[finger_idx])
    ws = []
    # grab the 4 indices surrounding the finger_idx:
    # [finger_idx - 2, finger_idx - 1, finger_idx, finger_idx + 1, finger_idx + 2]
    idxs = np.arange(-2, 3) + finger_idx
    for i in idxs:
        if 0 <= i < sensor_data.size:
            val = float(sensor_data[i])
            # weight is 1/(|xi - x| + 5)
            # kinda averaging it, but with a weighting
            ws += [1.0 / (np.abs(val - center_val) + 5.0)]
        else:
            ws += [0.0]

    ws = np.array(ws)
    finger_y = SENSOR_DIST * np.sum(ws * idxs) / np.sum(ws)

    return finger_y


# 4th degree polynomial
def quatric(x, c, d, e, f, g):
    return c*x**4 + d*x**3 + e*x**2 + f*x + g


def fit(sensor_data):
    sensor_data, rmved_idxs = remove_bad_data(sensor_data)
    ydata = np.array(sensor_data)
    sensors = SensorData.get_sensors()
    xdata = np.delete(sensors, rmved_idxs)

    popt = None
    rmse = np.inf
    peaks_min = None
    peaks_max = None
    if xdata.size > 3:
        # try to fit a 4th degree poly to data
        popt, residuals, rank, sing_vals, rcond = np.polyfit(xdata, ydata, 4, full=True)
        ypred = quatric(xdata, *popt)
        # find rms error
        rmse = np.sqrt(np.square(ydata - ypred).mean())

        # find approximate fitted curve and fins rel min and rel max
        # append extra predicted values beyond the 10 sensors in the front and back
        mod_sensors = np.arange(-EXTRA_LEN, NUM_SENSORS + EXTRA_LEN) * SENSOR_DIST
        fitted = quatric(mod_sensors, *popt)
        peaks_min = scipy.signal.argrelmin(fitted)[0]
        peaks_max = scipy.signal.argrelmax(fitted)[0]

        # delete values at the "edges"; there cannot be a pinch at
        # the edges
        peaks_max = np.delete(peaks_max, np.where(peaks_max >= fitted.size-EXTRA_LEN-1))
        peaks_max = np.delete(peaks_max, np.where(peaks_max <= EXTRA_LEN+1))

        # adjust indices to original sensor values
        peaks_min -= EXTRA_LEN
        peaks_max -= EXTRA_LEN
        # remove negative indices
        peaks_min = np.delete(peaks_min, np.where(peaks_min < 0))
        peaks_max = np.delete(peaks_max, np.where(peaks_max < 0))

    # max of points instead of curve
    # peaks_max = scipy.signal.argrelmax(sensor_data, order=4)[0]

    return popt, rmse, peaks_min, peaks_max


def detect(sensor_data):
    popt, rmse, peaks_min, peaks_max = fit(sensor_data)

    possible_gesture = "none"
    fingers = []
    if popt is not None and rmse < 15.0:
        sensors = SensorData.get_sensors()
        f = quatric(sensors, *popt)

        num_fingers = 0
        if len(peaks_max) == 0:
            possible_gesture = "swipe"
            num_fingers = 1
        elif len(peaks_max) == 1:
            possible_gesture = "pinch"
            num_fingers = 2

        if num_fingers != 0 and peaks_min.size >= num_fingers:
            # find lowest num_fingers amount of min_peaks
            min_idxs = f[peaks_min].argsort()[:num_fingers]
            finger_idxs = peaks_min[min_idxs]
            for idx in finger_idxs:
                fingers += [(f[idx], find_finger_y(idx, sensor_data))]

    return possible_gesture, fingers
