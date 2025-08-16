
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from utils.helper import split_data,bootstrap_test,eval_summary
import joblib
from utils.imputer import Dharma_Imputer




df = pd.read_excel('../data_curation/dataset_complications.xlsx')
feat_imp = ['Nausea','Loss_of_Appetite', 'Neutrophil_Percentage','Peritonitis','Ketones_in_Urine','Free_Fluids','CRP','WBC_Count','Body_Temperature','Appendix_Diameter']
feat_flag= ['Appendix_Diameter']
feat_model = ['Nausea','Loss_of_Appetite','Peritonitis','Ketones_in_Urine','Free_Fluids','CRP','WBC_Count','Body_Temperature','Appendix_Diameter']

df_imputer=pd.read_excel('../data_curation/dataset_model.xlsx')


df_imputer = df_imputer[feat_imp]
train_imp, val_imp , _  = split_data(df_imputer)
train_full_imp = pd.concat([train_imp, val_imp], axis=0, ignore_index=True)


imputer = Dharma_Imputer(feat_flag=feat_flag,feat_model=feat_model)
imputer.fit(train_full_imp)

df_train, df_val, df_test = split_data(df)

x_test = df_test[feat_imp]
y_test = df_test['Severity']

x_imputed = imputer.transform(x_test)

# x_imputed.to_excel('comp_test.xlsx', index = False)



dharma_comp = joblib.load('Dharma_comp.joblib')
metrics = [
    'AUC_ROC',
    'Accuracy',
    'Sensitivity',
    'Specificity',
    'PPV',
    'NPV'   
]

results = bootstrap_test( x_test = x_imputed, y_test = y_test, threshold= 0.3, model = dharma_comp, n_bootstraps = 5555)   

summary = eval_summary(df=results,metrics=metrics)

summary.to_excel('test_performance_rf.xlsx',index=False)





