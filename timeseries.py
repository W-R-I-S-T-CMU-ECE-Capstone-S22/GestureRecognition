import matplotlib.pyplot as plt
import numpy as np
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

from sktime.datasets import (
    load_arrow_head,
    load_basic_motions,
    load_japanese_vowels,
    load_plaid,
)


vowel_X_train, vowel_y_train = load_japanese_vowels(split="train")
vowel_X_test, vowel_y_test = load_japanese_vowels(split="test")

print(vowel_X_train, "\n")
print(vowel_y_train)

#print(type(vowel_X))

'''
plt.title(" First two dimensions of two instances of Japanese vowels")
plt.plot(vowel_X.iloc[0, 0])
plt.plot(vowel_X.iloc[1, 0])
plt.plot(vowel_X.iloc[0, 1])
plt.plot(vowel_X.iloc[1, 1])
plt.show()
'''

vowel_X_train.iloc[0, 0]
vowel_X_train.iloc[1, 0]
vowel_X_train.iloc[0, 1]
vowel_X_train.iloc[1, 1]

vowel_X_test.iloc[0, 0]
vowel_X_test.iloc[1, 0]
vowel_X_test.iloc[0, 1]
vowel_X_test.iloc[1, 1]

'''
motions_X, motions_Y = load_basic_motions(return_type="numpy3d")
motions_train_X, motions_train_y = load_basic_motions(
    split="train", return_type="numpy3d"
)
motions_test_X, motions_test_y = load_basic_motions(split="test", return_type="numpy3d")
print(type(motions_train_X))
print(
    motions_train_X.shape,
    motions_train_y.shape,
    motions_test_X.shape,
    motions_test_y.shape,
)
plt.title(" First and second dimensions of the first instance in BasicMotions data")
plt.plot(motions_train_X[0][0])
plt.plot(motions_train_X[0][1])


'''

#below doesn't work on unequal length of data

'''
rocket = RocketClassifier()
hc2 = HIVECOTEV2(time_limit_in_minutes=1)

rocket.fit(vowel_X_train, vowel_y_train)
y_pred = rocket.predict(vowel_X_test)
print(accuracy_score(vowel_y_test, y_pred))

hc2.fit(vowel_X_train, vowel_y_train)
y_pred = hc2.predict(vowel_X_test)
print(accuracy_score(vowel_y_test, y_pred))
'''

'''
steps = [
    ("concatenate", ColumnConcatenator()),
    ("classify", DrCIF(n_estimators=10)),
]
clf = Pipeline(steps)
clf.fit(vowel_X_train, vowel_y_train)
clf.score(vowel_X_test, vowel_y_test)
'''