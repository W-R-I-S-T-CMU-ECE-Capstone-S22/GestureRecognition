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
    files = ["data/" + filename for filename in os.listdir("data/") if filename != ".DS_Store" and not os.path.isdir("data/"+filename)]
    # files += ["live_data/swipe_two_med0.txt", "live_data/swipe_two_slow0.txt", "live_data/swipe_repeated_two.txt"]
    files += ["live_data/swipe_careless0.txt", "live_data/swipe_para_slow.txt", "live_data/swipe_para_slow0.txt"]

    datas = []
    ys = []
    for filename in files:
        sensor_datas = SensorDatasFromFile(filename)
        xs = sensor_datas.raw
        for x in xs:
            if np.count_nonzero(x == 255) > x.size - 2:
                continue
                # ys += [-1]
            else:
                if "one" in filename or "swipe" in filename:
                    ys += [0]
                elif "two" in filename:
                    ys += [1]
                elif "noise" in filename:
                    continue
                    # ys += [-1]

            x = preprocessing.scale(x)
            datas += [x]

    datas = np.array(datas)
    ys = np.array(ys)
    print(ys)
    X, y = datas, ys

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

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

    with open(model.MODEL_NAME, 'wb') as file:
        pickle.dump(clf, file)
