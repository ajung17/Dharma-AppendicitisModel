import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from Pipeline.pipeline import Pipeline_Diagnosis
from utils.helper import split_data, threshold_tune, optimize_sensi, tune_threshold
from sklearn.metrics import confusion_matrix


feat_flag = ['Appendix_Diameter']
df = pd.read_excel('../data_curation/dataset_model.xlsx')

train_df, val_df, _ = split_data (df=df)

Dharma = Pipeline_Diagnosis(strategy=None, feat_flag=feat_flag)

x_train = train_df.drop(columns=['Diagnosis'])
y_train = train_df['Diagnosis']

no_usg_df = val_df[val_df['Appendix_Diameter'].isna()]
x_nousg = no_usg_df.drop(columns=['Diagnosis'])
y_nousg = no_usg_df['Diagnosis']


x_val = val_df.drop(columns=['Diagnosis'])
y_val = val_df['Diagnosis']

print(x_val.isna().sum())

Dharma.fit(x_train,y_train)
y_prob = Dharma.predict_proba(x_val)[:, 1] 

# threshold = 0.64
# threshold = 0.44

# results = threshold_tune(y_true=y_nousg, y_prob=y_prob)
results = tune_threshold(y_true = y_val, y_prob = y_prob, mode= 'specificity', min_other=0.85) #(0.74)
# results = tune_threshold(y_true = y_val, y_prob = y_prob, mode= 'sensitivity', min_other=0.85) #(0.58)


print(results)



# print("Best Youden threshold:", results['best_youden'][0])  # threshold
# print("Youden J value:", results['best_youden'][1])         # J value

# print("Best Euclidean threshold:", results['best_euclidean'][0])  # threshold
# print("Euclidean distance:", results['best_euclidean'][1]) 

# y_pred = Dharma.predict(x_nousg) # default threshold for high sensi (97), spe(80)
# y_pred = (y_prob >= threshold).astype(int)  # eucledean distance threshold speci(96) sensi (89)

# tn, fp, fn, tp = confusion_matrix(y_nousg, y_pred).ravel()

# sensitivity = tp / (tp + fn)
# specificity = tn / (tn + fp)

# print("Sensitivity:", sensitivity)
# print("Specificity:", specificity)

# results = optimize_sensi(y_true=y_nousg, y_prob=y_prob, )

# print(results)
