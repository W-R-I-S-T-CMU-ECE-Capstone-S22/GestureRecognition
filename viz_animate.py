import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import sys
import time
import json
import random
import pickle
import paho.mqtt.client as mqtt

from sensor_data import SensorData, SensorDatasFromFile
import finger
import gesture
import model

from constants import *


y = SensorData.get_sensors()

fig = plt.figure()
axs = plt.axes(xlim=(0, 275), ylim=(-SENSOR_DIST, (NUM_SENSORS+1)*SENSOR_DIST))
scat = axs.scatter([], [])
title = axs.text(1.0, 1.0, "")
scat_fingers = axs.scatter([], [])

if len(sys.argv) <= 1:
    print("usage: python3 vz_webapp.py <filepath of data>")
    sys.exit(-1)

datas = SensorDatasFromFile(sys.argv[1])
timestamps = datas.timestamps
datas = datas.raw


def init():
    return scat, title, scat_fingers


def animate(i):
    if len(datas) > 0:
        sensor_data = datas.pop(0)

        x = sensor_data
        if x.size != y.size:
            return scat, title, scat_fingers,

        gest, fingers = finger.detect(x)

        print(gest, timestamps[i], time.time())

        title.set_text(f"Gesture={gest}")

        scat.set_offsets(np.array([x, y]).T)

        finger_xs = [x for x, y in fingers]
        finger_ys = [y for x, y in fingers]
        scat_fingers.set_offsets(np.array([finger_xs, finger_ys]).T)

        webapp_data = {}
        webapp_data["gesture"] = gest
        webapp_data["x_coord"] = finger_xs
        webapp_data["y_coord"] = finger_ys
        webapp_data["timestamp"] = timestamps[i]
        webapp_data = json.dumps(webapp_data)

        client.publish(GESTURE_TOPIC, webapp_data)
    return scat, title, scat_fingers,


client = mqtt.Client(
    "client" + str(random.randrange(100000, 999999)), clean_session=True)

client.connect("mqtt.eclipseprojects.io", 1883, 60)

client.loop_start()

anim = FuncAnimation(fig, animate, init_func=init, interval=33, blit=True)

plt.show()
plt.close("all")

client.loop_stop()
client.disconnect()
