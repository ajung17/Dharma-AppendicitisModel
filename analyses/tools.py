import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import pandas as pd
from utils.ss import calculate_scores

df = pd.read_excel('../data_curation/dataset_tools_imputed.xlsx')

systems = ['as','pas','air','tzanaki']
df_calculated = calculate_scores(df,systems) 

df_calculated.to_excel('tools.xlsx', index=False)



       
        
   
        


        



 
            


            








    

        
        