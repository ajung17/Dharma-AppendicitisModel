import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import joblib

from feat_complication.models import Models_Complications
from utils.helper import split_data

models = Models_Complications()

dharma = models.get_model('rf')
dharma_imputer = joblib.load('../../models/Dharma_Imputer.joblib')

df_base = pd.read_excel('../../data_curation/dataset_complications.xlsx')
feat_model = ['Nausea','Loss_of_Appetite','Peritonitis','Body_Temperature','WBC_Count','Neutrophil_Percentage','CRP','Ketones_in_Urine','Appendix_Diameter','Free_Fluids','Severity']
df_model = df_base[feat_model]

_,val, test = split_data(df_model)

df_train = pd.read_excel('../data_train_imputed.xlsx')

x_val = val.drop(columns='Severity')
y_val = val['Severity']

x_val_imputed = dharma_imputer.transform(x_val)
val_imputed = pd.DataFrame(x_val_imputed, columns=x_val.columns, index=x_val.index)
val_imputed['Severity'] = y_val.values

train = pd.concat([df_train,val_imputed], ignore_index=True)

x_train = train.drop(columns= 'Severity')
y_train = train ['Severity']

dharma.fit(x_train,y_train)

joblib.dump(dharma, 'dharma_comp.joblib')


