"""
See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.argrelmin.html
"""
import numpy as np
import scipy.signal
import scipy.optimize
import matplotlib.pyplot as plt


def remove_bad_data(sensor_data):
    idxs = np.where(sensor_data >= 175)
    new_data = np.delete(sensor_data, idxs)
    return new_data, idxs


# 4th degree polynomial
def quatric(x, c, d, e, f, g):
    return c*x**4 + d*x**3 + e*x**2 + f*x + g


def fit(sensor_data):
    sensor_data, rmved_idxs = remove_bad_data(sensor_data)
    ydata = np.array(sensor_data)
    sensor_nums = np.arange(10)
    xdata = np.delete(sensor_nums, rmved_idxs)

    popt = None
    rmse = np.inf
    peaks_min = None
    peaks_max = None
    if xdata.size > 3:
        # try to fit a 4th degree poly to data
        popt = np.polyfit(xdata, ydata, 4)
        ypred = quatric(xdata, *popt)
        # find rms error
        rmse = np.sqrt(np.square(ydata - ypred).mean())

        # find approximate fitted curve and fins rel min and rel max
        fitted = quatric(sensor_nums, *popt)
        peaks_min = scipy.signal.argrelmin(fitted)[0]
        peaks_max = scipy.signal.argrelmax(fitted)[0]

        # delete values at the "edges"; there cannot be a pinch at
        # the edges
        peaks_max = np.delete(peaks_max, np.where(peaks_max == fitted.size-2))
        peaks_max = np.delete(peaks_max, np.where(peaks_max == 1))

    return popt, rmse, peaks_min, peaks_max


def detect(sensor_data):
    min_idxs = scipy.signal.argrelmax(np.array(sensor_data), order=2)
    values = []
    for mini in min_idxs[0]:
        values.append((sensor_data[mini], mini))
    return values
