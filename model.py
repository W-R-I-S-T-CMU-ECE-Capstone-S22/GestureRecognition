import pickle
import numpy as np
from sklearn import preprocessing

MODEL_NAME = "models/model.pkl"
MODEL_NAME_ARM = "models/arm_model.pkl"

model_name = MODEL_NAME
clf = pickle.load(open(model_name, "rb"))

def predict(sensor_data):
    sensor_data = preprocessing.scale(sensor_data)
    return clf.predict([sensor_data])[0]

def pred2label(pred):
    if pred == -1:
        return "none"
    elif pred == 0:
        return "swipe"
    elif pred == 1:
        return "two"
    else:
        return "unknown"

def pred2num_fingers(pred):
    return pred + 1

