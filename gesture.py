# given ten readings of ten sensors, detect what gesture is happening
# all inputs are given as tuples
#
# rotating best cases:
# x x x x input1 x x x x x
# x x x x input2 x x x x x
# where input1 and input2 are less than x
# rotate left: input1[0] < input2[0]
# rotate right: input1[0] > input2[0]
#
# zooming best cases:
# x inputa2 x x x x x inputb2 x x
# x x x x inputa1 inputb1 x x x x
# where inputa1, inputa2, inputb1, and inputb2 are less than x
# zoom in: inputa2[0] < inputa1[0] && inputa2[1] < inputa1[1] && inputb1[0] < inputb2[0] && inputb1[1] < inputb2[1]
#
# x inputa1 x x x x x inputb1 x x
# x x x x inputa2 inputb2 x x x x
# where inputa1, inputa2, inputb1, and inputb2 are less than x
# zoom out: inputa1[0] < inputa2[0] && inputa1[1] < inputa2[1] && inputb2[0] < inputb1[0] && inputb2[1] < inputb1[1]
#

import numpy as np
import math
import finger


def classify(sensor_datas):
    # create frequency table of potential gestures
    # TO-DO: if there is a divide in type of gesture (i.e. 2 swipes, 2 pinches, 1 none), then classify as none? or look at previous gesture
    predictions = dict()
    for point in sensor_datas:
        possible_gesture, fingers = finger.detect(point)
        if possible_gesture in predictions:
            predictions[possible_gesture] += 1
        else:
            predictions[possible_gesture] = 1
    sort_predictions = sorted(
        predictions.items(), key=lambda x: x[1], reverse=True)
    predicted_gesture = sort_predictions[0][0]
    return predicted_gesture
