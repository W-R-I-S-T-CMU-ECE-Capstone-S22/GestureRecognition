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
    if "noise" in filename:
        correct_gesture = "none"

    xs = data.raw
    y = SensorData.get_sensors()

    correct = 0.0
    total = 0.0


    # print("-- new new detection --")
    for x in xs:
        total += 1
        prediction, fingers = finger.detect(x)
        pred = gesture.classify1(fingers)
        if np.count_nonzero(x == 255) > 2.0/3.0 * x.size:
            if pred in "none":
                correct += 1
        elif pred in correct_gesture:
            correct += 1

    new_new_det_accuracy = correct/total * 100

    correct = 0.0
    total = 0.0
    multiple_data = []
    n = 3
    # print("-- new detection --")
    for x in xs:
        prediction, _ = finger.detect(x)
        # print("data prediction: ", prediction)
        multiple_data.append(x)
        if len(multiple_data) > n:
            pred = gesture.classify(multiple_data)
            # print(pred)
            multiple_data = []

            total += 1
            if np.count_nonzero(x == 255) > 2.0/3.0 * x.size:
                if pred in "none":
                    correct += 1
            elif pred in correct_gesture:
                correct += 1

    new_det_accuracy = correct/total * 100

    print("Percent correct (new):", new_new_det_accuracy, "%")
    print("Percent correct:", new_det_accuracy, "%")
