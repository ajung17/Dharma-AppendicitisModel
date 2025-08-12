import pandas as pd

def scoring_systems(df, systems):

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
                scores += (df['Ipsilateral_Rebound_Tenderness'] == 1).astype(int) * 2
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
