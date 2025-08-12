import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from utils.helper import split_data

df=pd.read_excel('../data_curation/dataset_model.xlsx')
train_df, _, _ = split_data(df)

x= train_df.drop(columns=['Diagnosis'])
y= train_df['Diagnosis']

cv= StratifiedKFold(n_splits=10, shuffle=True, random_state=88)

# params_grid_1 = {
#     "n_estimators": [111, 333, 555],
#     "max_depth": [3, 7, 10],
#     "min_child_weight": [3, 6, 9],
#     "learning_rate": [0.01, 0.1, 0.3]
# }

params_grid_2 = {
    "n_estimators": [555],
    "max_depth": [3],
    "min_child_weight": [6],
    "learning_rate": [0.01],
    "gamma": [0, 1, 5],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha": [0, 0.1, 1],
    "reg_lambda": [1, 5, 10]
}

model = XGBClassifier(random_state=17)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=params_grid_2,
    scoring='roc_auc',    
    cv=cv,                 
    n_jobs=-1,
    verbose=1
)

grid_search.fit(x, y)

results_df = pd.DataFrame(grid_search.cv_results_)
results_df = results_df.sort_values(by="mean_test_score", ascending=False)

# print("Best Parameters:", grid_search.best_params_)
# print("Best CV AUC:", grid_search.best_score_)

print("Best Parameters for grid 2:", grid_search.best_params_)
print("Best CV AUC for grid 2:", grid_search.best_score_)

final_grid_xgboost= {
    "n_estimators": 555,
    "max_depth": 3,
    "min_child_weight": 6,
    "learning_rate": 0.01,
    "gamma": 0,
    "subsample": 0.6,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.1,
    "reg_lambda": 1
}
