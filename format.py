import os
import numpy as np
import pandas as pd

from sensor_data import SensorDatasFromFile


def formatFile(name):
    files = ["arm_data/" + str(name) + "/" + filename for filename in os.listdir(
        "arm_data/" + str(name) + "/") if filename != ".DS_Store" and not os.path.isdir("arm_data/" + str(name) + "/" + filename)]

    dataDict = dict()

    for i in range(10):
        dataDict['dim_'+str(i)] = []

    n = 20

    arr = np.array([])

    for filename in files:
        sensor_datas = SensorDatasFromFile(filename)
        xs = sensor_datas.raw
        fileLength = len(xs)-len(xs) % n
        for i in range(10):
            featureDict = dict()
            for x in range(0, fileLength):
                featureDict[x % n] = xs[x][i]
                if x % n == n-1:
                    newFeature = pd.Series(featureDict)
                    dataDict['dim_'+str(i)].append(newFeature)
                    featureDict = dict()
                    if i == 9:
                        if 'swipe_left' in filename:
                            arr = np.append(arr, 2)
                        elif 'swipe_right' in filename:
                            arr = np.append(arr, 1)
                        elif 'noise' in filename:
                            arr = np.append(arr, 0)
                        elif 'pinch_in' in filename:
                            arr = np.append(arr, -1)
                        elif 'pinch_out' in filename:
                            arr = np.append(arr, -2)

    formattedData = pd.DataFrame(dataDict)
    return formattedData, arr
