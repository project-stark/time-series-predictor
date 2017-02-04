import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder


class TimeSeriesPredictor:
    def __init__(self, entity_id):
        self.entity_id = entity_id
        self.clf = DecisionTreeRegressor()
        self.encoder = LabelEncoder()

    def fit(self, data):
        data = pd.DataFrame(data)
        data[["state"]] = self.encoder.fit_transform(np.ravel(data[["state"]]))
        self.clf.fit(data[["timestamp"]], data[["state"]])

    def predict(self, arg):
        return self.encoder.inverse_transform(self.clf.predict(pd.Series(arg)))

    def __repr__(self):
        return "Time Series Predictor for device {}".format(self.entity_id)
