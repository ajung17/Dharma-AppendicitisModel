import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from xgboost import XGBClassifier

data = pd.read_excel('../data_train_imputed.xlsx')

xgb = XGBClassifier(random_state=17)

# params_grid_1 = {
#     "n_estimators": [111, 333, 555,777],
#     "max_depth": [3, 7, 10],
#     "min_child_weight": [3, 6, 9],
#     "learning_rate": [0.01, 0.1, 0.3],
#     "scale_pos_weight": [1.5, 3, 5, 7]
#     }

params_grid_2 = {
    "n_estimators": [111],
    "max_depth": [3],
    "min_child_weight": [3],
    "learning_rate": [0.01],
    "gamma": [0, 1, 5],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha": [0, 0.1, 1],
    "reg_lambda": [1, 5, 10],
    "scale_pos_weight": [1.5, 3, 5, 7]

}

scoring = {
    'Recall': 'recall',
    'AUC': 'roc_auc',
    'AUPR': 'average_precision'
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=17)

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=params_grid_2,
    scoring=scoring, 
    refit='Recall',
    cv=cv,
    verbose=2,
    n_jobs=-1
)

x = data.drop(columns='Severity')
y = data['Severity']

grid_search.fit(x, y)

results = pd.DataFrame(grid_search.cv_results_)
results = results[[
    'rank_test_Recall', 
    'mean_test_Recall', 'std_test_Recall',
    'mean_test_AUC', 'std_test_AUC',
    'mean_test_AUPR', 'std_test_AUPR',
    'params'
]]

results_sorted = results.sort_values(by='mean_test_Recall', ascending=False)
results_sorted.to_excel('xgb_params_2.xlsx', index=False)