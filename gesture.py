import numpy as np
import math
import finger


def classify1(fingers, min_hist_len=3):
    if len(fingers) == 0:
        return "none"
    elif len(fingers) == 1:
        return "swipe"
    elif len(fingers) == 2:
        return "pinch"

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

