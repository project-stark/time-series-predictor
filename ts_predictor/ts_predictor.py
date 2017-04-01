"""ts_predictor.py: Machine learning models to do predictions on Time series data"""

import numpy as np
from sklearn.neural_network import MLPClassifier


class TimeSeriesPredictor:

    def __init__(self, datetimes, states):

        self._features = np.array([[dt.hour, dt.minute] for dt in datetimes]).reshape(len(datetimes), 2)
        self._labels = states

        self._clf = MLPClassifier()
        self._clf.fit(self._features, self._labels)

    def predict(self, datetime_obj):

        data = np.array([[datetime_obj.hour, datetime_obj.minute]])
        return self._clf.predict(data)[0]
