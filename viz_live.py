import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import sys
import time
import json
import random
import pickle
import paho.mqtt.client as mqtt

from sensor_data import SensorData, BatteryInfo
import finger
import model

from constants import *


y = SensorData.get_sensors()

fig = plt.figure()
axs = plt.axes(xlim=(0, 275), ylim=(-SENSOR_DIST, (NUM_SENSORS+1)*SENSOR_DIST))
scat = axs.scatter([], [])
title = axs.text(1.0, 1.0, "")
scat_fingers = axs.scatter([], [])

datas = []


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected!")
        client.subscribe(DATA_TOPIC)
        client.subscribe(BATT_TOPIC)

def on_disconnect(client, userdata, rc):
    if rc == 0:
        print("Disconnected!")
        client.loop_stop()

def on_message(client, userdata, msg):
    data = msg.payload
    if msg.topic == DATA_TOPIC:
        sensor_data = SensorData(data)

        datas.append(sensor_data)

def init():
    return scat, title, scat_fingers

def animate(i):
    if len(datas) > 0:
        sensor_data = datas.pop(0)

        x = sensor_data.raw
        if x.size != y.size:
            return scat, title, scat_fingers,

        gest, fingers = finger.detect(x)

        title.set_text(f"Gesture={gest}")

        scat.set_offsets(np.array([x, y]).T)

        finger_xs = [float("{0:.2f}".format(x)) for x, y in fingers]
        finger_ys = [float("{0:.2f}".format(y)) for x, y in fingers]
        scat_fingers.set_offsets(np.array([finger_xs, finger_ys]).T)
        print(gest, sensor_data.timestamp, time.time(), finger_xs, finger_ys)

        webapp_data = {}
        webapp_data["gesture"] = gest
        webapp_data["x_coord"] = finger_xs
        webapp_data["y_coord"] = finger_ys
        webapp_data["timestamp"] = sensor_data.timestamp
        webapp_data = json.dumps(webapp_data)

        client.publish(GESTURE_TOPIC, webapp_data)
    return scat, title, scat_fingers,

client = mqtt.Client(
    "client" + str(random.randrange(100000, 999999)), clean_session=True)
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_message = on_message

client.connect("172.26.52.57", 1883, 60)

client.loop_start()

anim = FuncAnimation(fig, animate, init_func=init, interval=10, blit=True)

plt.show()
plt.close("all")

client.disconnect()

