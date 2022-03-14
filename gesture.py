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
#

import numpy as np
import math
import finger


def identify(sensors):
    first_min = min(sensors)
    first_index = sensors.index(first_min)
    second_min = sorted(set(sensors))[1]
    second_index = sensors.index(second_min)
    minimums = []
    minimums.append((first_min, first_index))
    if (abs(first_index-second_index) > 2):
        minimums.append((second_min, second_index))
    return minimums


def classify(sensor1, sensor2):
    #fingers1 = finger.detect(sensor1)
    #fingers2 = finger.detect(sensor2)
    fingers1 = identify(sensor1)
    fingers2 = identify(sensor2)
    gesture = ""
    twoFingers = []
    # this is a weak case to distinguish between rotate and zoom. two data points for zoom indicate
    # two fingers, so the relative minimum will return the same number. need to define threshold for closeness.
    if (len(fingers1) == 1) and (len(fingers2) == 1):
        if(fingers1[0] < fingers2[0]):
            gesture = "rotateLeft"
        elif (fingers1[0] > fingers2[0]):
            gesture = "rotateRight"
        else:
            gesture = "noGesture"
    else:
        if (len(fingers1) == 2) and (len(fingers2) == 1):
            gesture = "zoomOut"
        elif (len(fingers2) == 2) and (len(fingers1) == 1):
            gesture = "zoomIn"
        else:
            gesture = "noGesture"
        # didn't account for two finger identification
        '''
        if ((fingers2[0] < fingers1[0]) and (fingers2[1] < fingers1[1]) and (fingers1[0] < fingers2[0]) and (fingers1[1] < fingers2[1])):
            gesture = "zoomIn"
        elif ((fingers1[0] < fingers2[0]) and (fingers1[1] < fingers2[1]) and (fingers2[0] < fingers1[0]) and (fingers2[1] < fingers1[1])):
            gesture = "zoomOut"
        else:
            gesture = "noGesture"
        '''
    return (gesture, fingers1, fingers2)
