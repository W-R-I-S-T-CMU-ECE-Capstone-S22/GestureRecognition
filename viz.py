"""
Visualizes sensor data and predicted finger position.

Run code and continually press x on plots.

Predicted fingers will show up as red squares.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

from sensor_data import SensorData
from gesture import GestureRecognizer

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("usage: python3 vix.py <filepath of data>")
        sys.exit(-1)

    data = SensorData(sys.argv[1])

    xs = data.raw
    y = np.arange(10)

    for x in xs:
        vals, idxs = GestureRecognizer.identify(x)
        for i in idxs[0]:
            plt.plot(x[i], i, "sr")
        plt.scatter(x, y)
        plt.gca().set_xlim(left=0, right=275)
        plt.show()

