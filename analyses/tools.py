import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_excel('../data_curation/dataset_tools_imputed.xlsx')


def scoring_systems(df, columns, systems):

    for system in systems:

    
        if system == 'as':  
            scores_as = pd.Series(0, index=df.index)
        
        
            for col in columns:
                if col == 'Lower_Right_Abd_Pain':
                    scores_as += df[col].apply(lambda x: 2 if x == 1 else 0)
                elif col == 'WBC_Count':
                    scores_as += df[col].apply(lambda x: 2 if x >= 10 else 0)
                elif col in ['Migratory_Pain', 'Loss_of_Appetite', 'Nausea', 'Ipsilateral_Rebound_Tenderness']:
                    scores_as += df[col].apply(lambda x: 1 if x == 1 else 0)
                elif col == 'Body_Temperature':
                    scores_as += df[col].apply(lambda x: 1 if x >= 37.3 else 0)
                elif col == 'Neutrophil_Percentage':
                    scores_as += df[col].apply(lambda x: 1 if x >= 75 else 0)
                else:
                    pass
            
            df['Alvarado_score'] = scores_as
            return df
        
        if system == 'pas':
            scores_pas = pd.Series(0, index= df.index)

            for col in columns:
                if col == 'Lower_Right_Abd_Pain':
                    scores_pas += df[col].apply(lambda x: 2 if x == 1 else 0)
                elif col == 'Coughing_Pain':
                    scores_pas += df[col].apply(lambda x: 2 if x == 1 else 0)
                elif col in ['Migratory_Pain', 'Loss_of_Appetite', 'Nausea']:
                    scores_pas += df[col].apply(lambda x: 1 if x == 1 else 0)
                elif col == 'WBC_Count':
                    scores_pas += df[col].apply(lambda x: 1 if x >= 10 else 0)
                elif col == 'Body_Temperature':
                    scores_pas += df[col].apply(lambda x: 1 if x >= 38 else 0)
                elif col == 'Neutrophil_Percentage':
                    scores_pas += df[col].apply(lambda x: 1 if x >= 75 else 0)
                else:
                    pass

            df['PAS_score'] = scores_pas
            return df
        
        if system == 'air':
            scores_air = pd.Series(0, index= df.index)

            for col in columns:
                if col in ['Migratory_Pain','Nausea']:
                    scores_air += df[col].apply(lambda x: 1 if x == 1 else 0)
                elif col == 'Ipsilateral_Rebound_Tenderness':
                    scores_air += df[col].apply(lambda x: 2 if x == 1 else 0)
                elif col == 'CRP':
                    scores_air += df[col].apply(lambda x: 0 if x < 10 else (1 if x < 50 else 2))
                elif col == 'Neutrophil_Percentage':
                    scores_air += df[col].apply(lambda x: 0 if x < 70 else (1 if x < 85 else 2))
                elif col == 'WBC_Count':
                    scores_air += df[col].apply(lambda x: 0 if x < 10 else (1 if x < 15 else 2))
                elif col == 'Body_Temperature':
                    scores_air += df[col].apply(lambda x: 1 if x >= 38.5  else 0)
                else:
                    pass
                
            df['AIR_score'] = scores_air
            return df
            
        if system == 'tzanaki':
            scores_tzanaki = pd.Series(0, index= df.index)

            for col in columns:
                if col == 'Ipsilateral_Rebound_Tenderness':
                    scores_tzanaki += df[col].apply(lambda x: 3 if x == 1 else 0)
                elif col == 'Lower_Right_Abd_Pain':
                    scores_tzanaki += df[col].apply(lambda x: 4 if x == 1 else 0)
                elif col == 'WBC_Count':
                    scores_tzanaki += df[col].apply(lambda x: 2 if x > 12  else 0)
                elif col == 'Appendix_Diameter':
                    scores_tzanaki += df[col].apply(lambda x: 6 if x >= 6 else 0 )
                else:
                    pass

            df['Tzanaki_score']= scores_tzanaki
            return df
    
    return df


        



 
            


            








    

        
        