from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import logging
from sklearn.impute import SimpleImputer as SkSimpleImputer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)

class Simple_Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, feat_continuous=None, feat_categorical=None, feat_model=None, feat_flag=None, placeholder=-1):
        self.placeholder = placeholder
        self.feat_flag = feat_flag

        self.feat_continuous = feat_continuous if feat_continuous is not None else ['WBC_Count', 'Neutrophil_Percentage', 'CRP', 'Body_Temperature']
        self.feat_categorical = feat_categorical if feat_categorical is not None else ['Nausea', 'Loss_of_Appetite', 'Peritonitis',
                                                                                       'Ketones_in_Urine', 'Free_Fluids']
        self.feat_model = feat_model if feat_model is not None else ['Nausea', 'Loss_of_Appetite', 'Peritonitis', 'WBC_Count',
                                                                    'Neutrophil_Percentage', 'CRP', 'Ketones_in_Urine',
                                                                    'Appendix_Diameter', 'Free_Fluids']
        if self.feat_flag is not None:
            self.feat_model += [f"{col}_flag" for col in self.feat_flag]
            self.feat_model = list(dict.fromkeys(self.feat_model))

        self.imputer_continuous = SkSimpleImputer(strategy='mean')
        self.imputer_categorical = SkSimpleImputer(strategy='most_frequent')

        logging.info(f"Initialized Simple_Imputer with mean for continuous and mode for categorical.")

    def missing_flag(self, X):
        X_copy = X.copy()
        if self.feat_flag is not None:
            for col in self.feat_flag:
                X_copy[f"{col}_flag"] = X_copy[col].isna().astype(int)
                X_copy[col] = X_copy[col].fillna(self.placeholder)
                logging.info(f"Missing flag set for column: {col}")
        return X_copy

    def fit(self, X, y=None):
        X = self.missing_flag(X) if self.feat_flag is not None else X
        self.imputer_continuous.fit(X[self.feat_continuous])
        self.imputer_categorical.fit(X[self.feat_categorical])
        logging.info("Simple_Imputer fit completed.")
        return self

    def transform(self, X, y=None):
        X = self.missing_flag(X) if self.feat_flag is not None else X
        X_copy = X.copy()
        X_copy[self.feat_continuous] = self.imputer_continuous.transform(X_copy[self.feat_continuous])
        X_copy[self.feat_categorical] = self.imputer_categorical.transform(X_copy[self.feat_categorical])
        X_copy = X_copy[self.feat_model] if self.feat_model is not None else X_copy
        logging.info("Simple_Imputer transform completed.")
        return X_copy
