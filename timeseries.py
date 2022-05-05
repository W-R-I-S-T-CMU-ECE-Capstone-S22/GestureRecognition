import matplotlib.pyplot as plt
import numpy as np
import format
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.dictionary_based import TemporalDictionaryEnsemble
from sktime.classification.hybrid import HIVECOTEV2
from sktime.classification.interval_based import DrCIF
from sktime.classification.kernel_based import RocketClassifier
from sktime.transformations.panel.compose import ColumnConcatenator
from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor


def timeseriesModel():
    trainingData, trainingResult = format.formatFile("train")
    testingData, testingResult = format.formatFile("test")

    rocket = RocketClassifier()
    rocket.fit(trainingData, trainingResult)
    y_pred = rocket.predict(testingData)
    score = accuracy_score(testingResult, y_pred)

    print(score)

    '''
    hc2 = HIVECOTEV2(time_limit_in_minutes=1)
    hc2.fit(trainingData, trainingResult)
    y_pred = rocket.predict(testingData)
    score = accuracy_score(testingResult, y_pred)

    print(score)


    steps = [
        ("concatenate", ColumnConcatenator()),
        ("classify", DrCIF(n_estimators=10)),
    ]
    clf = Pipeline(steps)
    clf.fit(trainingData, trainingResult)
    print(clf.score(testingData, testingResult))

    clf = ColumnEnsembleClassifier(
        estimators=[
            ("DrCIF0", DrCIF(n_estimators=10), [0]),
            ("TDE3", TemporalDictionaryEnsemble(max_ensemble_size=5), [3]),
        ]
    )
    clf.fit(trainingData, trainingResult)
    print(clf.score(testingData, testingResult))
    '''

    return


timeseriesModel()
