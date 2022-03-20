import sys
import numpy as np

import finger
import gesture
from sensor_data import SensorData, SensorDatasFromFile


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("usage: python3 test.py <filepath of data>")
        sys.exit(-1)

    filename = sys.argv[1]
    data = SensorDatasFromFile(filename)

    correct_gesture = filename.replace("data/", "").replace(".txt", "")
    # remove numbers
    correct_gesture = ''.join([i for i in correct_gesture if not i.isdigit()])

    xs = data.raw
    y = SensorData.get_sensors()

    correct = 0.0
    total = 0.0
    for x in xs:
        if np.count_nonzero(x == 255) > 2.0/3.0 * x.size:
            continue
        pred_gesture, fingers = finger.detect(x)
        print(pred_gesture, correct_gesture)
        if pred_gesture in correct_gesture:
            correct += 1

        total += 1

    print("Percent correct:", correct/total * 100, "%")

