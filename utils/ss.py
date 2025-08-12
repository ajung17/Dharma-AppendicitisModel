import pandas as pd
from sklearn.metrics import  recall_score, precision_score, roc_auc_score, accuracy_score
import numpy as np 
import logging

logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s — %(levelname)s — %(message)s"
)


def calculate_scores(df, systems):

    for system in systems:
        if system == 'as':
            scores = pd.Series(0, index=df.index)
            if 'Lower_Right_Abd_Pain' in df:
                scores += (df['Lower_Right_Abd_Pain'] == 1).astype(int) * 2
            if 'WBC_Count' in df:
                scores += (df['WBC_Count'] >= 10).astype(int) * 2
            for col in ['Migratory_Pain', 'Loss_of_Appetite', 'Nausea', 'Ipsilateral_Rebound_Tenderness']:
                if col in df:
                    scores += (df[col] == 1).astype(int)
            if 'Body_Temperature' in df:
                scores += (df['Body_Temperature'] >= 37.3).astype(int)
            if 'Neutrophil_Percentage' in df:
                scores += (df['Neutrophil_Percentage'] >= 75).astype(int)
            df['Alvarado_score'] = scores

        elif system == 'pas':
            scores = pd.Series(0, index=df.index)
            for col in ['Lower_Right_Abd_Pain', 'Coughing_Pain']:
                if col in df:
                    scores += (df[col] == 1).astype(int) * 2
            for col in ['Migratory_Pain', 'Loss_of_Appetite', 'Nausea']:
                if col in df:
                    scores += (df[col] == 1).astype(int)
            if 'WBC_Count' in df:
                scores += (df['WBC_Count'] >= 10).astype(int)
            if 'Body_Temperature' in df:
                scores += (df['Body_Temperature'] >= 38).astype(int)
            if 'Neutrophil_Percentage' in df:
                scores += (df['Neutrophil_Percentage'] >= 75).astype(int)
            df['PAS_score'] = scores

        elif system == 'air':
            scores = pd.Series(0, index=df.index)
            for col in ['Migratory_Pain', 'Nausea']:
                if col in df:
                    scores += (df[col] == 1).astype(int)
            if 'Ipsilateral_Rebound_Tenderness' in df:
                scores += (df['Ipsilateral_Rebound_Tenderness'] == 1).astype(int) * 3
            if 'CRP' in df:
                scores += df['CRP'].apply(lambda x: 0 if x < 10 else (1 if x < 50 else 2))
            if 'Neutrophil_Percentage' in df:
                scores += df['Neutrophil_Percentage'].apply(lambda x: 0 if x < 70 else (1 if x < 85 else 2))
            if 'WBC_Count' in df:
                scores += df['WBC_Count'].apply(lambda x: 0 if x < 10 else (1 if x < 15 else 2))
            if 'Body_Temperature' in df:
                scores += (df['Body_Temperature'] >= 38.5).astype(int)
            df['AIR_score'] = scores

        elif system == 'tzanaki':
            scores = pd.Series(0, index=df.index)
            if 'Ipsilateral_Rebound_Tenderness' in df:
                scores += (df['Ipsilateral_Rebound_Tenderness'] == 1).astype(int) * 3
            if 'Lower_Right_Abd_Pain' in df:
                scores += (df['Lower_Right_Abd_Pain'] == 1).astype(int) * 4
            if 'WBC_Count' in df:
                scores += (df['WBC_Count'] > 12).astype(int) * 2
            if 'Appendix_Diameter' in df:
                scores += (df['Appendix_Diameter'] >= 6).astype(int) * 6
            df['Tzanaki_score'] = scores

    return df


def tools_metrics(x, y, threshold, n_bootstraps = 5555):
    rng = np.random.RandomState(17)

    results=[]

    for i in range(n_bootstraps):
        logging.info(F'STARTING ON SAMPLE_NO : {i+1} ')
        boot_indices = rng.choice(len(y), size=len(y), replace=True)

        x_boot = x.iloc[boot_indices]
        y_boot = y.iloc[boot_indices]
        y_pred = (x_boot >= threshold).astype(int)


        auroc_score = roc_auc_score(y_boot, x_boot)
        ppv_score = precision_score(y_boot, y_pred, pos_label=1)
        npv_score = precision_score(y_boot, y_pred, pos_label=0)
        sensitivity_score = recall_score(y_boot, y_pred, pos_label=1)
        specificity_score = recall_score(y_boot, y_pred, pos_label=0)
        accuracy = accuracy_score(y_boot, y_pred)

        results.append({
            'sample': f'bootstrap_{i+1}',
            'AUC_ROC': auroc_score,
            'Accuracy': accuracy,
            'Sensitivity': sensitivity_score,
            'Specificity': specificity_score,
            'PPV': ppv_score,
            'NPV': npv_score
        })
        
        logging.info(f'BOOTSTRAPPING ON SAMPLE {i+1} ENDED')

    df=pd.DataFrame(results)

    return df

    
