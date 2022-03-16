"""
See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.argrelmin.html
"""
import numpy as np
import scipy.signal
import scipy.optimize
import matplotlib.pyplot as plt


def func_quad(x, a, b, c, d, e):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e


def func_double(x, a, b, c):
    return a*x**2 + b*x + c


def fit(sensor_data):
    ydata = np.array(sensor_data)
    xdata = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    popt1, _ = scipy.optimize.curve_fit(func_quad, xdata, ydata)
    popt2, _ = scipy.optimize.curve_fit(func_double, xdata, ydata)
    return popt1, popt2


def detect(sensor_data):
    min_idxs = scipy.signal.argrelmax(np.array(sensor_data), order=2)
    values = []
    for mini in min_idxs[0]:
        values.append((sensor_data[mini], mini))
    return values
