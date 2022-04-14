import pickle
import numpy as np
from sklearn import preprocessing

MODEL_NAME = "models/model_old.pkl"
MODEL_NAME_NORMALIZED = "models/model_normalized.pkl"
MODEL_NAME_OTHER = "models/model_other.pkl"
MODEL_NAME_SUBSETS = "models/model_subsets.pkl"
MODEL_NAME_ARM = "models/arm_model.pkl"

model_name = MODEL_NAME_OTHER
clf = pickle.load(open(model_name, "rb"))

def predict(sensor_data):
    if model_name == MODEL_NAME_NORMALIZED:
        sensor_data = preprocessing.scale(sensor_data)
        # sensor_data = (sensor_data - np.mean(sensor_data)) / (np.std(sensor_data) + 0.1)
    return clf.predict([sensor_data])[0]

def pred2label(pred):
    if pred == -1:
        return "none"
    elif pred == 0:
        return "swipe"
    elif pred == 1:
        return "pinch"
    else:
        return "unknown"

def pred2num_fingers(pred):
    return pred + 1
