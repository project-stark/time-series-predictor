import numpy as np
from sklearn.tree import DecisionTreeRegressor


class TimeSeriesPredictor:
    def __init__(self, data):
        states = list(map(lambda x: x["state"], data))
        self._classes = list(set(states))
        self._states = list(map(lambda state: self._classes.index(state), states))
        self._timestamps = list(map(lambda x: x["timestamp"], data))
        self._clf = DecisionTreeRegressor()

    def fit(self, data_point):
        if data_point["state"] not in self._classes:
            self._classes.append(data_point["state"])

        self._states.append(self._classes.index(data_point["state"]))
        self._timestamps.append(data_point["timestamp"])
        print(np.array(list(self._timestamps)).reshape(len(self._timestamps), 1))
        self._clf.fit(np.array(list(self._timestamps)).reshape(len(self._timestamps), 1), np.array(self._states))

    def predict(self, arg):
        return self._classes[int(self._clf.predict(np.array(arg)))]
