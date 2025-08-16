import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from utils.helper import split_data
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from utils.imputer import Dharma_Imputer
from utils.models import Models_Complications
from utils.helper import bootstrap, eval_summary
from sklearn.metrics import make_scorer, recall_score, precision_score
import joblib
from sklearn.model_selection import train_test_split



df = pd.read_excel('../data_curation/dataset_complications.xlsx')
feat_imp = ['Nausea','Loss_of_Appetite', 'Neutrophil_Percentage','Peritonitis','Ketones_in_Urine','Free_Fluids','CRP','WBC_Count','Body_Temperature','Appendix_Diameter']
feat_flag= ['Appendix_Diameter']
feat_model = ['Nausea','Loss_of_Appetite','Peritonitis','Ketones_in_Urine','Free_Fluids','CRP','WBC_Count','Body_Temperature','Appendix_Diameter']
feat_cat = ['Nausea','Loss_of_Appetite','Peritonitis','Ketones_in_Urine','Free_Fluids']


models = Models_Complications()
dharma = models.get_model(model_name='Dharma')

df_imputer=pd.read_excel('../data_curation/dataset_model.xlsx')

df_imputer = df_imputer[feat_imp]
train_imp, val_imp , _  = split_data(df_imputer)
train_full_df = pd.concat([train_imp, val_imp], axis=0, ignore_index=True)


imputer = Dharma_Imputer(feat_flag=feat_flag,feat_model=feat_imp)
imputer.fit(train_full_df)

df_train, df_val, df_test = split_data(df)
# df_imb = pd.concat([df_train,df_val],axis=0, ignore_index=True)

df_imb = df_train[feat_imp]

x_imb = df_imb[feat_imp]

x_imputed = imputer.transform(x_imb)
y_imb = df_train['Severity']




smote = SMOTE(sampling_strategy='auto', k_neighbors= 10, random_state=17)
# smote_enn = SMOTEENN(
#     sampling_strategy='auto',
#     random_state=17,
#     smote=smote
# )

print(f'Before SMOTE: {y_imb.value_counts()}')

x_smote, y_smote = smote.fit_resample(x_imputed, y_imb)
print (f'After SMOTE: {y_smote.value_counts()}')


majority_class_size = y_smote.value_counts()[0]
desired_majority_size = majority_class_size // 2  

nearmiss = NearMiss(version=1,sampling_strategy={0: desired_majority_size})
x_final, y_final = nearmiss.fit_resample(x_smote, y_smote)

# x_train,x_val,y_train,y_val = train_test_split(x_final,y_final,test_size=0.25,random_state=17)


for col in feat_cat:
    x_final[col] = x_final[col].round().astype(int)



print (f'After NearMiss : {y_final.value_counts()}')
print(x_final.dtypes)

specificity = make_scorer(recall_score, pos_label=0)
npv = make_scorer(precision_score, pos_label=0)

scoring = {
    'roc_auc': 'roc_auc',
    'accuracy': 'accuracy',
    'specificity': specificity,
    'npv': npv,
    'sensitivity': 'recall',
    'ppv': 'precision'
    
}

# metrics = {
#     'roc_auc',
#     'accuracy',
#     'specificity',
#     'npv',
#     'sensitivity',
#     'ppv'   
# }

# results = bootstrap( x_final, y_final, model=dharma, scoring=scoring,n_bootstraps = 555 )


# summary = eval_summary(results, metrics)
# summary.to_excel('complications_summary_np.xlsx', index=False)

dharma_comp = dharma.fit(x_final,y_final)


joblib.dump(dharma_comp, "Dharma_comp.joblib")
















