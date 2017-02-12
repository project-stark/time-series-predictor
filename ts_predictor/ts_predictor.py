"""ts_predictor.py: Machine learning models to do predictions on Time series data"""


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


class TimeSeriesPredictor:

    def __init__(self, datetimes, states):

        self._encoder = LabelEncoder()
        self._minutes_of_day = np.array(list(map(lambda dt: (dt.hour * 60) + dt.minute, datetimes)))
        self._states = self._encoder.fit_transform(states)

        self._clf = LogisticRegression(solver="sag", multi_class="multinomial", max_iter=10000, warm_start=True)

        self._clf.fit(self._minutes_of_day.reshape(len(self._minutes_of_day), 1), self._states)

    def predict(self, datetime_obj):

        return self._encoder.inverse_transform(self._clf.predict(np.array([(datetime_obj.hour * 60) +
                                                                           datetime_obj.minute]).reshape(1, -1)))[0]
