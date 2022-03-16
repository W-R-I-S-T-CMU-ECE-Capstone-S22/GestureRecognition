"""
Visualizes sensor data and predicted finger position.

Run code and continually press x on plots.

Predicted fingers will show up as red squares.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

from sensor_data import SensorData
import finger

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("usage: python3 viz.py <filepath of data>")
        sys.exit(-1)

    data = SensorData(sys.argv[1])

    xs = data.raw
    y = np.arange(10)

    for x in xs:
        popt1, popt2 = finger.fit(x)
        vals = finger.detect(x)
        for val in vals:
            plt.plot(val[0], val[1], "sr")
        plt.plot(finger.func_quad(y, *popt1), y, "orange")
        plt.plot(finger.func_double(y, *popt2), y, "green")
        plt.scatter(x, y)
        plt.gca().set_xlim(left=0, right=275)
        plt.show()
