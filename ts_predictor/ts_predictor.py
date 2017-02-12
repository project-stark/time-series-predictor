"""ts_predictor.py: Machine learning models to do predictions on Time series data"""


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


class TimeSeriesPredictor:

    def __init__(self, timestamps, states):

        self._encoder = LabelEncoder()
        self._timestamps = np.array(timestamps)
        self._states = self._encoder.fit_transform(states)

        self._clf = LogisticRegression(solver="sag", multi_class="multinomial", max_iter=10000, warm_start=True)

        self._clf.fit(self._timestamps.reshape(len(self._timestamps), 1), self._states)

    def predict(self, timestamp):

        return self._encoder.inverse_transform(self._clf.predict(np.array([timestamp]).reshape(1,-1)))[0]
