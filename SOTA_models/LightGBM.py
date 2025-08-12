import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from utils.helper import split_data
from lightgbm import LGBMClassifier


df=pd.read_excel('../data_curation/dataset_model.xlsx')
train_df, _, _ = split_data(df)

x= train_df.drop(columns=['Diagnosis'])
y= train_df['Diagnosis']

cv= StratifiedKFold(n_splits=10, shuffle=True, random_state=88)

# params_grid_1 = {
#     "n_estimators": [111, 333, 555],           
#     "max_depth": [3, 7, 10],                   
#     "min_child_samples": [3, 6, 9],            
#     "learning_rate": [0.01, 0.1, 0.3]          
# }

params_grid_2 = {
    "n_estimators": [333],
    "max_depth": [3],
    "min_child_samples": [3],
    "learning_rate": [0.01],
    "min_gain_to_split": [0, 1, 5],         
    "bagging_fraction": [0.6, 0.8, 1.0],    
    "feature_fraction": [0.6, 0.8, 1.0],   
    "lambda_l1": [0, 0.1, 1],                
    "lambda_l2": [1, 5, 10]
}

model = LGBMClassifier(random_state=17)

grid_search = GridSearchCV(
    estimator = model,
    param_grid = params_grid_2,
    scoring='roc_auc',    
    cv=cv,                 
    n_jobs=-1,
    verbose=1
)

grid_search.fit(x, y)

results_df = pd.DataFrame(grid_search.cv_results_)
results_df = results_df.sort_values(by="mean_test_score", ascending=False)

# print("Best Parameters for LGBM on grid 1:", grid_search.best_params_)
# print("Best CV AUC for LGBM on grid 1:", grid_search.best_score_)

print("Best Parameters for LGBM on grid 2:", grid_search.best_params_)
print("Best CV AUC for LGBM on grid 2:", grid_search.best_score_)

final_grid_lgbm = {
    "n_estimators": 333,
    "max_depth": 3,
    "min_child_samples": 3,
    "learning_rate": 0.01,
    "min_gain_to_split": 0,         
    "bagging_fraction": 0.6,    
    "feature_fraction": 0.6,   
    "lambda_l1": 0,                
    "lambda_l2": 1
}
