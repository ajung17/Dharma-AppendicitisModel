
from sklearn.experimental import enable_iterative_imputer 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

class Selective_Imputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns_iter = ['WBC_Count', 'Neutrophil_Percentage', 'CRP', 'Body_Temperature']
        self.columns_knn = ['Nausea', 'Loss_of_Appetite', 'Peritonitis',
                             'Ketones_in_Urine', 'Free_Fluids']
        
        self.imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=10, random_state=17),
        max_iter=10,
        random_state=17
        )

        self.imputer_knn = KNNImputer(n_neighbors=5)

    def fit(self, x, y=None):
        x_copy = x.copy()
        self.imputer.fit(x[self.columns_iter])
        x_copy[self.columns_iter] = self.imputer.transform(x[self.columns_iter])
        self.imputer_knn.fit(x_copy)
        return self

    def transform(self, x, y=None):
        x_copy = x.copy()
        x_copy[self.columns_iter] = self.imputer.transform(x_copy[self.columns_iter])
        x_copy = self.imputer_knn.transform(x_copy)
        df= pd.DataFrame(x_copy, columns=x.columns, index=x.index)
        df[self.columns_knn] = df[self.columns_knn].round().astype(int)
        return df
    





