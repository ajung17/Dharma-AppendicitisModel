import pandas as pd
import numpy as np
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor


class Imputer:
    def __init__(self, columns, estimator=None, max_iter=10, random_state=17):
        if estimator is None:
            estimator = RandomForestRegressor(n_estimators=10, random_state=random_state)

        self.imputer = IterativeImputer(
            estimator=estimator,
            max_iter=max_iter,
            random_state=random_state
        )
    
    def fit(self, x):
        self.imputer.fit(x)
        return
    
    def transform(self, x):
        x_copy= x.copy()
        x_copy[self.columns]= self.imputer.transform(x_copy[self.columns])
        return x_copy

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

        