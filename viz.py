"""
Visualizes sensor data and predicted finger position.

Run code and continually press x on plots.
"""
import sys
import pickle

import matplotlib.pyplot as plt
import numpy as np

from sensor_data import SensorData, SensorDatasFromFile
import finger
import gesture

from constants import *

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("usage: python3 viz.py <filepath of data>")
        sys.exit(-1)

    data = SensorDatasFromFile(sys.argv[1])

    xs = data.raw
    y = SensorData.get_sensors()
    mod_y = np.arange(-finger.EXTRA_LEN, NUM_SENSORS +
                      finger.EXTRA_LEN) * SENSOR_DIST

    clf = pickle.load(open(MODEL_NAME_OTHER, "rb"))
    for x in xs:
        popt, rmse, peaks_min, peaks_max = finger.fit(x)
        gest, fingers = finger.detect(x)
        gest = gesture.classify1(fingers)

        # print('err:', rmse)
        if popt is not None and rmse < 15.0:
            f = finger.quartic(mod_y, *popt)
            plt.plot(f, mod_y, "orange")

            for i in peaks_max:
                plt.plot(f[i], i*SENSOR_DIST, "sg")
                plt.text(f[i], i*SENSOR_DIST, "max")

            for i in peaks_min:
                plt.plot(f[i], i*SENSOR_DIST, "s", color="orange")
                plt.text(f[i], i*SENSOR_DIST, "min")

        for finger_x, finger_y in fingers:
            plt.plot(finger_x, finger_y, "<r")
            plt.text(finger_x, finger_y, "finger")

        print(clf.predict([x]))

        plt.scatter(x, y)
        plt.title(f"Gesture={gest}")
        plt.xlabel("distance (mm)")
        plt.ylabel("sensors (mm)")
        plt.gca().set_xlim(left=0, right=275)
        plt.show()
