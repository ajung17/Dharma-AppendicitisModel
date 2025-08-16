import sys
import os
root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from utils.helper import split_data
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier
from utils.imputer import Dharma_Imputer


df = pd.read_excel('../data_curation/dataset_complications.xlsx')
feat_imp =['Nausea','Loss_of_Appetite','Peritonitis','Body_Temperature','WBC_Count','Neutrophil_Percentage','CRP','Ketones_in_Urine','Appendix_Diameter','Free_Fluids']
feat_flag= ['Appendix_Diameter']

df_imputer=pd.read_excel('../data_curation/dataset_model.xlsx')


df_imputer = df_imputer[feat_imp]
train_imp, val_imp , _  = split_data(df_imputer)
train_full_imp = pd.concat([train_imp, val_imp], axis=0, ignore_index=True)


imputer = Dharma_Imputer(feat_flag=feat_flag,feat_model=feat_imp)
imputer.fit(train_full_imp)


model = XGBClassifier(random_state=17)

train, val, test = split_data(df)

x_train = train[feat_imp]
y_train = train['Severity']

x_train_imputed = imputer.transform(x_train)

neg_count = np.sum(y_train == 0)
pos_count = np.sum(y_train == 1)
balanced_weight = neg_count / pos_count

print(balanced_weight)


params_grid_1 = {
    "n_estimators": [111, 333, 555],
    "max_depth": [3, 7, 10],
    "min_child_weight": [3, 6, 9],
    "learning_rate": [0.01, 0.1, 0.3],
    "scale_pos_weight": [
        balanced_weight,
        balanced_weight * 2,
        balanced_weight * 3,
        balanced_weight * 4,
        balanced_weight * 5,
        balanced_weight * 6
    ]
}

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
    "scale_pos_weight": [
        balanced_weight,
        balanced_weight * 2,
        balanced_weight * 3,
        balanced_weight * 4,
        balanced_weight * 5,
        balanced_weight * 6
    ]
}

# final_grid = {
#     "n_estimators": [111],
#     "max_depth": [3],
#     "min_child_weight": [3],
#     "learning_rate": [0.01],
#     "gamma": [0],
#     "subsample": [0.6],
#     "colsample_bytree": [0.6],
#     "reg_alpha": [0],
#     "reg_lambda": [5],
#     "scale_pos_weight": [
#        
#         balanced_weight * 4,
#        
#        
#     ]

# }

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=17)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=params_grid_2,
    scoring='recall',    
    cv=cv,                 
    n_jobs=-1,
    verbose=1
    
)

grid_search.fit(x_train_imputed, y_train)

results_df = pd.DataFrame(grid_search.cv_results_)
results_df = results_df.sort_values(by="mean_test_score", ascending=False)

print(results_df)

# print("Best Parameters:", grid_search.best_params_)
# print("Best CV Recall:", grid_search.best_score_)

# final_grid = {
#     "n_estimators": [111],
#     "max_depth": [3],
#     "min_child_weight": [3],
#     "learning_rate": [0.01],
#     "gamma": [0],
#     "subsample": [0.6],
#     "colsample_bytree": [0.6],
#     "reg_alpha": [0],
#     "reg_lambda": [5],
#     "scale_pos_weight": [
#         balanced_weight,
#         balanced_weight * 2,
#         balanced_weight * 3,
#         balanced_weight * 4,
#         balanced_weight * 5,
#         balanced_weight * 6
#     ]

# }




