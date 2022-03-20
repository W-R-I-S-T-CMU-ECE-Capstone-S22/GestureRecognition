import sys
import time
import random
import paho.mqtt.client as mqtt

import matplotlib.pyplot as plt
import numpy as np

from sensor_data import SensorData, BatteryInfo
import finger

from constants import *

DATA_TOPIC = "wrist/data/sensors"
BATT_TOPIC = "wrist/batt/sensors"
BATT_TOPIC_ASK = "wrist/batt/ask"

NUM_SENSORS = 10

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
        print(pred_gesture)

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
        time.sleep(5)
        client.publish(BATT_TOPIC_ASK, 0)
    except KeyboardInterrupt:
        client.disconnect()
        time.sleep(0.1)
        break