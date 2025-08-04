
from sklearn.experimental import enable_iterative_imputer 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s — %(levelname)s — %(message)s"
)

class Dharma_Imputer(BaseEstimator, TransformerMixin):
    def __init__(self,col_cont=None, col_others=None, strategy_cont=None, strategy_others=None):
        self.columns_cont = col_cont if col_cont is not None else ['WBC_Count', 'Neutrophil_Percentage', 'CRP', 'Body_Temperature']
        self.columns_others = col_others if col_others is not None else ['Nausea', 'Loss_of_Appetite', 'Peritonitis',
                             'Ketones_in_Urine', 'Free_Fluids']

        if strategy_cont is None:
            self.imputer = IterativeImputer(
                max_iter=10,
                random_state=17,
            )           
        elif strategy_cont == 'rf':
            self.imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=10, random_state=17),
            max_iter=10,
            random_state=17
            )
        elif strategy_cont == 'linear':
            self.imputer = IterativeImputer(
                max_iter=10,
                random_state=17,
                estimator=LinearRegression()
            )
        else:
            raise ValueError("Invalid strategy for continuous imputation. Use 'rf', 'linear' or None for BayesianRidge.")

        if strategy_others is None:
            self.imputer_knn = KNNImputer(n_neighbors=5)
        elif strategy_others == 'logreg':
            self.imputer_knn= IterativeImputer(
                max_iter=10,
                random_state=17,
                estimator= LogisticRegression(max_iter=555)
            )
        else:
            raise ValueError("Invalid strategy for others imputation. Use 'logreg' or None for KNNImputer.")


        logging.info("Dharma_Imputer initialized with columns: %s for iterative imputation and %s for KNN imputation.", self.columns_cont, self.columns_others)

    def fit(self, x, y=None):
        x_copy = x.copy()
        self.imputer.fit(x[self.columns_cont])
        x_copy[self.columns_cont] = self.imputer.transform(x[self.columns_cont])
        self.imputer_knn.fit(x_copy)
        return self

    def transform(self, x, y=None):
        x_copy = x.copy()
        x_copy[self.columns_cont] = self.imputer.transform(x_copy[self.columns_cont])
        x_copy = self.imputer_knn.transform(x_copy)
        df= pd.DataFrame(x_copy, columns=x.columns, index=x.index)
        df[self.columns_others] = df[self.columns_others].round().astype(int)
        return df

def Selective_Impute(dataset, col_all=None, col_cont=None, strategy_cont=None, col_others=None, strategy_others=None):
    logging.info("Initial DataFrame shape: %s, Missing values: %s, Columns: %s, Dtypes: %s",
                 dataset.shape, dataset.isna().sum(), dataset.columns, dataset.dtypes)
    if col_all is not None:
        df_change = dataset[col_all]
        df_nochange = dataset.drop(columns=col_all)
    else:
        df_change = dataset.copy()
        df_nochange = pd.DataFrame()
    imputer = Dharma_Imputer(col_cont=col_cont, col_others=col_others, strategy_cont=strategy_cont, strategy_others=strategy_others)
    df_imputed = imputer.fit_transform(df_change)
    logging.info("Imputation completed. Imputed DataFrame shape: %s, Missing values: %s, Columns: %s, Dtypes: %s",
                 df_imputed.shape, df_imputed.isna().sum(), df_imputed.columns, df_imputed.dtypes)
    df_return = pd.concat([df_nochange, df_imputed], axis=1)
    logging.info("Final DataFrame shape after concatenation: %s", df_return.shape)
    logging.info("Final DataFrame columns: %s, Dtypes: %s", df_return.columns)

    return df_return[dataset.columns]


