
from os import name
from sklearn.experimental import enable_iterative_imputer 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s — %(levelname)s — %(message)s"
)

class Dharma_Imputer(BaseEstimator, TransformerMixin):
    def __init__(self,feat_continuous=None, feat_categorical=None, feat_model=None, feat_flag=None, placeholder= -1, strategy=None):

        self.placeholder = placeholder
        self.feat_flag = feat_flag
        self.strategy= strategy 
        self.feat_continuous = feat_continuous if feat_continuous is not None else ['WBC_Count', 'Neutrophil_Percentage', 'CRP', 'Body_Temperature']
        self.feat_categorical = feat_categorical if feat_categorical is not None else ['Nausea', 'Loss_of_Appetite', 'Peritonitis',
                             'Ketones_in_Urine', 'Free_Fluids']
        self.feat_model = feat_model if feat_model is not None else ['Nausea','Loss_of_Appetite','Peritonitis','WBC_Count','Neutrophil_Percentage','CRP','Ketones_in_Urine','Appendix_Diameter','Free_Fluids']
        if self.feat_flag:
            self.feat_model += [f"{col}_flag" for col in self.feat_flag]
            self.feat_model = list(dict.fromkeys(self.feat_model))
        self.dtc={}
        logging.info("Initializing Dharma_Imputer with strategy: %s, columns: %s", strategy, feat_continuous + feat_categorical)

        if self.strategy is None or self.strategy == 'dt':
            self.imputer_continuous = IterativeImputer(
                max_iter=55,
                random_state=17,
                estimator=DecisionTreeRegressor(max_depth=7),
                skip_complete=True
            )   
        elif self.strategy == 'linear':
            self.imputer_continuous = IterativeImputer(
                max_iter=55,
                random_state=17,
                estimator=LinearRegression(),
                skip_complete=True
            )
        elif self.strategy == 'knn':
            self.imputer_continuous = KNNImputer(n_neighbors=5)
        else:
            raise ValueError("Invalid strategy for continuous imputation. Use DecisionTreeRegressor: 'dt', LinearRegression: 'linear', KNNImputer: 'knn'.")
        logging.info("Imputer initialized with strategy: %s", self.strategy)

        self.imputer_categorical = DecisionTreeClassifier(max_depth=7, random_state=17)

    def missing_flag(self, x):
        x_copy = x.copy()
        for col in self.feat_flag:
            x_copy[f"{col}_flag"] = x_copy.loc[:,col].isna().astype(int)
            x_copy.loc[:,col] = x_copy[col].fillna(self.placeholder)
            logging.info(f"Missing flag set for column: {col}")
        return x_copy

    def fit(self, x, y=None):
        x = self.missing_flag(x) if self.feat_flag is not None else x
        x_copy = x.copy()
        self.imputer_continuous.fit(x_copy[self.feat_continuous])
        x_copy[self.feat_continuous] = self.imputer_continuous.transform(x_copy[self.feat_continuous])

        for col in self.feat_categorical:
            x_train = x_copy[x_copy[col].notna()].drop(columns=[col])
            y_train = x_copy.loc[x_copy[col].notna(), col]
            model = DecisionTreeClassifier(max_depth=7, random_state=17)
            model.fit(x_train, y_train)
            self.dtc[col] = model
            logging.info(f"Fitted classifier for: {col} with features: {x_train.columns.tolist()}")
        logging.info("Fitting Dharma_Imputer completed.")
        return self
        
    def transform(self, x, y=None):
        x = self.missing_flag(x) if self.feat_flag is not None else x
        x_copy = x.copy()
        x_copy[self.feat_continuous] = self.imputer_continuous.transform(x_copy[self.feat_continuous])
        
        for col in self.feat_categorical:
            model = self.dtc.get(col, None)
            missing_mask = x_copy[col].isna()
            if model is not None and missing_mask.any():
                to_predict = x_copy.loc[missing_mask].drop(columns=[col])
                x_copy.loc[missing_mask, col] = model.predict(to_predict)
            logging.info(f"Transformed column: {col} with features: {to_predict.columns.tolist()}.")
        logging.info("Transforming data with Dharma_Imputer completed.")
        x_copy = x_copy[self.feat_model] if self.feat_model is not None else x_copy
        return x_copy
