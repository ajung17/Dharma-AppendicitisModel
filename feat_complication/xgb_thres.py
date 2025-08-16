import sys
import os
root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from utils.helper import split_data, optimize_sensi
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
    n_estimators=111,
    max_depth=3,
    min_child_weight=3,
    learning_rate=0.01,
    gamma=0,
    subsample=0.6,
    colsample_bytree=0.6,
    reg_alpha=0,
    reg_lambda=5,
    scale_pos_weight=11,
    random_state=17
)


train, val, test = split_data(df)

x_train = train[feat_imp]
y_train = train['Severity']

x_val = val[feat_imp]
y_val = val['Severity']

x_val_imputed = imputer.transform(x_val)
x_train_imputed = imputer.transform(x_train)

model.fit(x_train_imputed,y_train)
y_prob = model.predict_proba(x_val_imputed)[:, 1]

result = optimize_sensi(y_val, y_prob, min_specificity=0.65)

print (result)
 













