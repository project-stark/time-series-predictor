import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder


class TimeSeriesPredictor:
    def __init__(self, entity_id):
        self._entity_id = entity_id
        self._clf = DecisionTreeRegressor()
        self._encoder = LabelEncoder()

    def fit(self, data):
        df = pd.DataFrame(data)
        df[["state"]] = self._encoder.fit_transform(np.ravel(df[["state"]]))
        self._clf.fit(df[["timestamp"]], df[["state"]])

    def predict(self, arg):
        result = self._clf.predict(pd.Series(arg).values.reshape(-1, 1))
        int_result = np.array(result, dtype="int32")
        return self._encoder.inverse_transform(int_result)[0]

    def __repr__(self):
        return "Time Series Predictor for device {}".format(self._entity_id)
