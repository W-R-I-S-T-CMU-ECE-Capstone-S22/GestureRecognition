import sys
import time
import json
import random
import paho.mqtt.client as mqtt

import matplotlib.pyplot as plt
import numpy as np

from sensor_data import SensorData, BatteryInfo
import finger

from constants import *

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

        pred_gesture, fingers = finger.detect(sensor_data.raw)

        webapp_data = {}
        webapp_data["gesture"] = pred_gesture
        webapp_data["x_coord"] = [x for x,y in fingers]
        webapp_data["y_coord"] = [y for x,y in fingers]
        webapp_data["timestamp"] = time.time()
        webapp_data = json.dumps(webapp_data)

        print(webapp_data)
        client.publish(GESTURE_TOPIC, webapp_data)

    elif msg.topic == BATT_TOPIC:
        batt = BatteryInfo(data)
        print("voltage:", batt.voltage, "| battery (%):", batt.percentage)


client = mqtt.Client("client" + str(random.randrange(100000, 999999)), clean_session=True)
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_message = on_message

client.connect("mqtt.eclipseprojects.io", 1883, 60)

client.loop_start()

while (1):
    try:
        time.sleep(10)
        client.publish(BATT_TOPIC_ASK, 0)
    except KeyboardInterrupt:
        client.disconnect()
        time.sleep(0.1)
        break

