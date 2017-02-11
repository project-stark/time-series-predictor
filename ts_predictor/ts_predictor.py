import numpy as np
from sklearn.linear_model import LogisticRegression


class TimeSeriesPredictor:
    def __init__(self, data):
        states = data.values()
        self._classes = list(set(states))
        self._states = list(map(lambda state: self._classes.index(state), states))
        self._timestamps = list(data.keys())
        self._clf = LogisticRegression(solver="sag", multi_class="multinomial", max_iter=10000, warm_start=True)
        self._clf.fit(np.array(self._timestamps).reshape(len(self._timestamps), 1), np.array(self._states))

    def predict(self, timestamp):
        return self._classes[int(self._clf.predict(np.array(timestamp)))]
