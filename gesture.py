import numpy as np
import math
import finger

history = []


def classify1(fingers, min_hist_len=3):
    global history

    if len(fingers) == 0:
        history = []
        return "none"

    if len(history) < min_hist_len:
        history += [fingers]
        return "none"

    prev_fingers = history[-min_hist_len:]
    history += [fingers]

    num_swipes = 0
    num_pinches = 0
    for prevs in prev_fingers:
        if len(prevs) == 1:
            num_swipes += 1
        else:
            num_pinches += 1

    if len(fingers) == 1:
        curr, prev = np.array(fingers[0]), np.array(prev_fingers[-1][0])

        dx, dy = np.abs(curr - prev)
        # if np.linalg.norm((dx,dy)) > 15:
        #    print("delta too big!", curr, prev)
        #    return "none"

        if num_swipes > num_pinches:
            return "swipe"
        else:
            return "pinch"
    else:
        return "pinch"

    return "none"


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
