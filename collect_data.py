import sys
import time
import random
import paho.mqtt.client as mqtt

LOG = None
if len(sys.argv) > 1:
    LOG = sys.argv[1]

DATA_TOPIC = "wrist/data/sensors"
BATT_TOPIC = "wrist/batt/sensors"
BATT_TOPIC_ASK = "wrist/batt/ask"

NUM_SENSORS = 10
OUT_FILE_NAME = "data.txt" if LOG is None else LOG

if LOG: out_file = open(OUT_FILE_NAME, "w")

def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))

    client.subscribe(DATA_TOPIC)
    client.subscribe(BATT_TOPIC)

def on_disconnect(client, userdata, rc):
    print("Disconnected with result code " + str(rc))

    client.loop_stop()

def on_message(client, userdata, msg):
    data = list(msg.payload)
    if msg.topic == DATA_TOPIC:
        timestamp = int.from_bytes(bytes(data[:-NUM_SENSORS]), "little")
        curr_time = time.time()
        elapsed = curr_time - timestamp

        sensor_data = data[-NUM_SENSORS:]
        print(elapsed, ":\t", sensor_data)

        if LOG: out_file.write(f"{curr_time}, {sensor_data}\n")
    elif msg.topic == BATT_TOPIC:
        timestamp, voltage, battery = msg.payload.decode("utf-8").split(",")
        timestamp = int(timestamp)
        voltage = float(voltage)
        battery = float(battery)

        print("voltage:", voltage, "| battery (%):", battery)

client = mqtt.Client("client" + str(random.randrange(100000, 999999)), clean_session=True)
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_message = on_message

client.connect("172.26.52.57", 1883, 60)

client.loop_start()

while (1):
    try:
        time.sleep(10)
        client.publish(BATT_TOPIC_ASK, 0)
    except KeyboardInterrupt:
        if LOG: out_file.close()
        break
