import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import sys
import time
import json
import random
import pickle
import socket
import threading
import paho.mqtt.client as mqtt

from sensor_data import SensorData, BatteryInfo
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

datas = []

run = True

HOST = "172.26.165.98"  # The Photons's hostname or IP address
PORT = 23           # The port used by the Photon

def get_data():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)

    while True:
        if not run:
            break
        try:
            print("Connecting...")
            s.connect((HOST, PORT))
            print("Connected!")
            while True:
                if not run:
                    break
                try:
                    data = s.recv(1 + NUM_SENSORS)
                    sensor_data = SensorData(data)
                    datas.append(sensor_data)
                except:
                    break
            s.close()
        except:
            print("Connection lost! Retrying...")
            time.sleep(1)

    print("Data collection stopped!")

def init():
    t = threading.Thread(target=get_data)
    t.start()

    return scat, title, scat_fingers

def animate(i):
    if len(datas) > 0:
        sensor_data = datas.pop(0)

        x = sensor_data.raw
        if x.size != y.size:
            return scat, title, scat_fingers,

        _, fingers = finger.detect(x)
        gest = gesture.classify(fingers)

        print(gest, sensor_data.timestamp, time.time())

        label = model.pred2label(model.predict(x))

        title.set_text(f"Gesture={label}")

        scat.set_offsets(np.array([x, y]).T)

        finger_xs = [x for x, y in fingers]
        finger_ys = [y for x, y in fingers]
        scat_fingers.set_offsets(np.array([finger_xs, finger_ys]).T)

        webapp_data = {}
        webapp_data["gesture"] = gest
        webapp_data["x_coord"] = finger_xs
        webapp_data["y_coord"] = finger_ys
        webapp_data["timestamp"] = sensor_data.timestamp
        webapp_data = json.dumps(webapp_data)

        client.publish(GESTURE_TOPIC, webapp_data)
    return scat, title, scat_fingers,


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT!")
        client.subscribe(DATA_TOPIC)
        client.subscribe(BATT_TOPIC)


def on_disconnect(client, userdata, rc):
    if rc == 0:
        print("Disconnected from MQTT!")
        client.loop_stop()

client = mqtt.Client(
    "client" + str(random.randrange(100000, 999999)), clean_session=True)
client.on_connect = on_connect
client.on_disconnect = on_disconnect
# client.on_message = on_message

client.connect("mqtt.eclipseprojects.io", 1883, 60)

client.loop_start()

anim = FuncAnimation(fig, animate, init_func=init, interval=16, blit=True)

plt.show()
plt.close("all")

run = False
client.disconnect()
