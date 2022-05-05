"""
Visualizes sensor data and predicted finger position.

Run code and continually press x on plots.
"""
import sys
import json
import random
import struct
import paho.mqtt.client as mqtt

import matplotlib.pyplot as plt
import numpy as np

from sensor_data import SensorData, SensorDatasFromFile
import finger
import model

from constants import *

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("usage: python3 viz.py <filepath of data>")
        sys.exit(-1)

    data = SensorDatasFromFile(sys.argv[1])

    client = mqtt.Client(
        "client" + str(random.randrange(100000, 999999)), clean_session=True)
    client.connect("mqtt.eclipseprojects.io", 1883, 60)
    client.loop_start()

    xs = data.raw
    timestamps = data.timestamps
    y = SensorData.get_sensors()

    for i,x in enumerate(xs):
        gest, fingers = finger.detect(x)

        for finger_x, finger_y in fingers:
            plt.plot(finger_x, finger_y, "<r")
            plt.text(finger_x, finger_y, "finger")

        finger_xs = [x for x, y in fingers]
        finger_ys = [y for x, y in fingers]

        webapp_data = {}
        webapp_data["gesture"] = gest
        webapp_data["x_coord"] = finger_xs
        webapp_data["y_coord"] = finger_ys
        webapp_data["timestamp"] = timestamps[i]
        webapp_data = json.dumps(webapp_data)

        client.publish(GESTURE_TOPIC, webapp_data)
        client.publish(DATA_TOPIC,
                       struct.pack('%sb' % x.size, *(x.astype(np.int8))))


        plt.scatter(x, y)
        plt.title(f"Gesture={gest}")
        plt.xlabel("distance (mm)")
        plt.ylabel("sensors (mm)")
        plt.gca().set_xlim(left=0, right=275)
        plt.show()

    client.disconnect()

