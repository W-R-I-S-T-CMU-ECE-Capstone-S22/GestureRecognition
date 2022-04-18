import time
import pickle
import numpy as np
import scipy.signal
import scipy.optimize

import model

from constants import *
from sensor_data import SensorData


EXTRA_LEN = 1


# 4th degree polynomial
def quartic(x, a, b, c, d, e):
    return a*x**4 + b*x**3 + c*x**2 + d*x**1 + e*x**0


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
            # averaging it, but with a weighting
            ws += [1 / (np.abs(val - center_val) + 5)]
        else:
            ws += [1 / 5]

    ws = np.array(ws)
    finger_y = SENSOR_DIST * np.sum(ws * idxs) / np.sum(ws)

    return finger_y


def fit(sensor_data):
    ydata, rmved_idxs = remove_bad_data(sensor_data)
    sensors = SensorData.get_sensors()
    xdata = np.delete(sensors, rmved_idxs)

    popt = None
    rmse = np.inf
    peaks_min = None
    peaks_max = None
    if xdata.size > 3 and xdata.size == ydata.size:
        # try to fit a 4th degree poly to data
        popt, residuals, rank, sing_vals, rcond = np.polyfit(
            xdata, ydata, 4, full=True)
        ypred = quartic(xdata, *popt)
        # find rms error
        rmse = np.sqrt(np.square(ydata - ypred).mean())

        # find approximate fitted curve and fins rel min and rel max
        # append extra predicted values beyond the 10 sensors in the front and back
        mod_sensors = np.arange(-EXTRA_LEN, NUM_SENSORS +
                                EXTRA_LEN) * SENSOR_DIST
        fitted = quartic(mod_sensors, *popt)
        peaks_min = scipy.signal.argrelmin(fitted)[0]
        peaks_max = scipy.signal.argrelmax(fitted)[0]

        # delete values at the "edges"; there cannot be a pinch at
        # the edges
        peaks_max = np.delete(peaks_max, np.where(
            peaks_max >= fitted.size-EXTRA_LEN-2))
        peaks_max = np.delete(peaks_max, np.where(peaks_max <= EXTRA_LEN+2))

        # adjust indices to original sensor values
        peaks_min -= EXTRA_LEN
        peaks_max -= EXTRA_LEN
        # remove negative indices
        peaks_min = np.delete(peaks_min, np.where(peaks_min < 0))
        peaks_max = np.delete(peaks_max, np.where(peaks_max < 0))

    return popt, rmse, peaks_min, peaks_max

def detect(sensor_data):
    pred = model.pred2num_fingers(model.predict(sensor_data))
    popt, rmse, peaks_min, peaks_max = fit(sensor_data)

    pred_gesture = "none"
    fingers = []
    if popt is not None and rmse < 15.0:
        sensors = SensorData.get_sensors()
        f = quartic(sensors, *popt)

        if pred == 1:
            pred_gesture = "swipe"
            if peaks_min.size == 1:
                min_idx = f[peaks_min].argsort()[0]
                idx = peaks_min[min_idx]
            else:
                idx = np.argmin(f)

            finger = np.array((f[idx], find_finger_y(idx, sensor_data)))
            fingers = [finger]

        elif pred == 2:
            pred_gesture = "two"
            idx = np.argmin(f)
            finger = np.array((f[idx], np.mean(SensorData.get_sensors())))
            fingers = [finger]

        # if pred == 3:
            # pred_gesture = "all"
            # f, _ = remove_bad_data(f)
            # x = np.mean(f)
            # y = np.mean(SensorData.get_sensors())
            # fingers = [(x, y)]

    return pred_gesture, fingers

