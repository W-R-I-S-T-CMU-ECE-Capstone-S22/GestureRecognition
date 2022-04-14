import os
import numpy as np
import pickle

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from sensor_data import SensorData, SensorDatasFromFile
import model

if __name__ == '__main__':
    files = ["arm_data/" + filename for filename in os.listdir("arm_data/") if filename != ".DS_Store" and not os.path.isdir("arm_data/"+filename)]
    # files += ["new/noise.txt"]

    datas = []
    ys = []
    for filename in files:
        sensor_datas = SensorDatasFromFile(filename)
        xs = sensor_datas.raw
        for x in xs:
            if np.count_nonzero(x == 255) > 2.0/3.0 * x.size:
                ys += [-1]
            else:
                if "swipe" in filename:
                    ys += [0]
                elif "pinch" in filename:
                    ys += [1]
                elif "noise" in filename:
                    ys += [-1]

            # x = preprocessing.scale(x)
            #x = (x - np.mean(x)) / (np.std(x) + 0.1)
            datas += [x]

    datas = np.array(datas)
    ys = np.array(ys)
    X, y = datas, ys

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0)

    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            SVC(), tuned_parameters, scoring='%s_macro' % score,
            verbose=5
        )
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

    with open(model.MODEL_NAME_ARM, 'wb') as file:
        pickle.dump(clf, file)
