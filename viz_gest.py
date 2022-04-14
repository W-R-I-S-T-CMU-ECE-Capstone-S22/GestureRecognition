"""
Visualizes sensor data and predicted finger position.

Run code and continually press x on plots.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

from sensor_data import SensorData, SensorDatasFromFile
import finger, gesture

from constants import *

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("usage: python3 viz.py <filepath of data>")
        sys.exit(-1)

    data = SensorDatasFromFile(sys.argv[1])

    xs = data.raw
    y = SensorData.get_sensors()
    mod_y = np.arange(-finger.EXTRA_LEN,NUM_SENSORS+finger.EXTRA_LEN) * SENSOR_DIST

    finger_locs = []
    for i,x in enumerate(xs):
        gest, fingers = finger.detect(x)
        gest = gesture.classify(fingers)
        for f in fingers:
            plt.scatter(f[0], f[1])
            plt.text(f[0], f[1], f"t={i}")

        plt.title(f"Gesture={gest}")
        plt.xlabel("distance (mm)")
        plt.ylabel("sensors (mm)")
        plt.gca().set_xlim(left=0, right=275)
        plt.gca().set_ylim(bottom=0, top=NUM_SENSORS*SENSOR_DIST)
        plt.show()
