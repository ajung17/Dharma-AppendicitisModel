import sys
import os
root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from utils.helper import split_data,bootstrap_test, eval_summary
import pandas as pd
import joblib
import numpy as np
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


model = XGBClassifier(
    n_estimators=333,
    max_depth=3,
    min_child_weight=3,
    learning_rate=0.01,
    gamma=0,
    subsample=0.6,
    colsample_bytree=0.6,
    reg_alpha=0,
    reg_lambda=5,
    scale_pos_weight=7,
    random_state=17
)


train, val, test = split_data(df)

# train_final = pd.concat([train,val], ignore_index=True)

x_train = train[feat_imp]
y_train = train['Severity']

x_test = test[feat_imp]
y_test = test['Severity']

x_test_imputed = imputer.transform(x_test)
x_train_imputed = imputer.transform(x_train)

model.fit(x_train_imputed,y_train)
y_prob = model.predict_proba(x_test_imputed)[:, 1]

metrics = [
    'AUC_ROC',
    'Accuracy',
    'Sensitivity',
    'Specificity',
    'PPV',
    'NPV'   
]

results = bootstrap_test( x_test = x_test_imputed, y_test = y_test, threshold= 0.55, model = model, n_bootstraps = 5555)   

summary = eval_summary(df=results,metrics=metrics)

summary.to_excel('test_performance_woval.xlsx',index=False)