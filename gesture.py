from re import S
import numpy as np
import math
import finger

recentGestures = ['none' for i in range(5)]

currentGesture = 'none'


def classify(fingers, min_hist_len=3):
    if len(fingers) == 0:
        return "none"
    elif len(fingers) == 1:
        return "swipe"
    elif len(fingers) == 2:
        return "two"

# create frequency table of number of fingers and finds highest frequency


def findHighestFrequency(sensor_datas):
    predictions = dict()
    for point in sensor_datas:
        possible_gesture, _ = finger.detect(point)
        if possible_gesture in predictions:
            predictions[possible_gesture] += 1
        else:
            predictions[possible_gesture] = 1
    sort_predictions = sorted(
        predictions.items(), key=lambda x: x[1], reverse=True)
    predicted_gesture = sort_predictions[0][0]
    predicted_gesture_list = sort_predictions
    return predicted_gesture, predicted_gesture_list


def classifyOnFrequency(predictedNumber):
    possibleGesture = ''
    if predictedNumber == 0:
        possibleGesture = 'none'
    elif predictedNumber == 1:
        possibleGesture = 'swipe'
    elif predictedNumber == 2:
        possibleGesture = 'pinch'
    recentGestures.pop(0)
    recentGestures.append(possibleGesture)

    predictions = dict()
    for gesture in recentGestures:
        if gesture in predictions:
            predictions[gesture] += 1
        else:
            predictions[gesture] = 1
    sort_predictions = sorted(
        predictions.items(), key=lambda x: x[1], reverse=True)

    predictionString = ''

    for gest in recentGestures:
        if gest == 'pinch':
            predictionString += 'p'
        elif gest == 'swipe':
            predictionString += 's'
        elif gest == 'none':
            predictionString += 'n'

    if predictionString.find('pppp') == 0:
        if predictionString[-1] == 's':
            return 'pinchIn'
        elif predictionString == 'ppppp':
            return 'pinch'
        else:
            return currentGesture

    elif predictionString.find('pppp') == 1:
        if predictionString[0] == 's':
            return 'pinchOut'
        else:
            return currentGesture

    if 'ppp' in predictionString:
        if 'ss' in predictionString:
            return 'pinchOut'
        else:
            return 'pinchIn'

    if 'pp' in predictionString:
        if predictionString.find('sss') == 0:
            return 'pinchOut'
        else:
            return 'pinchIn'

    if sort_predictions[0][1] >= 4:
        return sort_predictions[0][0]

    return currentGesture
