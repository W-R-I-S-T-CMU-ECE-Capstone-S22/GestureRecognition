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
        popt, rmse, peaks_min, peaks_max = finger.fit(x)

        # print('err:', rmse)
        if popt is not None and rmse < 15.0:
            f = finger.quatric(y, *popt)
            plt.plot(f, y, "orange")

            for i in peaks_min:
                plt.plot(f[i], i, "sr")
            for i in peaks_max:
                plt.plot(f[i], i, "sg")

            if len(peaks_max) == 1:
                plt.title("pinch")
            elif len(peaks_max) == 0:
                plt.title("swipe")
        else:
            plt.title("none")

        plt.scatter(x, y)
        plt.gca().set_xlim(left=0, right=275)
        plt.show()
