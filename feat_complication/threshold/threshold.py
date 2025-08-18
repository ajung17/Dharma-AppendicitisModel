import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from feat_complication.models import Models_Complications
import joblib
from utils.helper import split_data,bootstrap_test,eval_summary,tune_threshold



data = pd.read_excel('../data_train_imputed.xlsx')
x_train = data.drop(columns= 'Severity')
y_train = data['Severity']

models = Models_Complications()

rf = models.get_model(model_name='rf')

rf.fit(x_train, y_train)

df_base = pd.read_excel('../../data_curation/dataset_complications.xlsx')
feat_model = ['Nausea','Loss_of_Appetite','Peritonitis','Body_Temperature','WBC_Count','Neutrophil_Percentage','CRP','Ketones_in_Urine','Appendix_Diameter','Free_Fluids','Severity']
df_base = df_base[feat_model]

_, val, _ = split_data(df_base)

x_val = val.drop(columns = 'Severity')
y_val = val['Severity']

imputer = joblib.load('../../models/Dharma_Imputer.joblib')

x_val_imputed = imputer.transform(x_val)

y_prob = rf.predict_proba(x_val_imputed)[:,1]

# results = bootstrap_test(x_val_imputed,y_val, rf)

# threshold = tune_threshold(y_true= y_val, y_prob=y_prob, mode='sensitivity', min_other=0.55)


# metrics = [
#     'AUC_ROC',
#     'Accuracy',
#     'Sensitivity',
#     'Specificity',
#     'PPV',
#     'NPV'   
# ]

# summary = eval_summary(results,metrics)

# summary.to_excel('val_performance.xlsx', index=False)

# print (threshold) (0.44)

# print(y_val.value_counts())

