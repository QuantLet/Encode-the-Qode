import os
import json
import pickle
import pandas as pd
from IPython.display import display
from torch.utils.data import Dataset
import numpy as np

def parse_meta(row):
    row = row['metainfo_file']
    if row=='empty':
        return ['','','','']
    dict_keys = list(row.keys())
    dict_key_n = [k.lower() for k in dict_keys]
    name_idx = np.where(['name' in k for k in dict_key_n])[0]
    desc_idx = np.where(['desc' in k for k in dict_key_n])[0]
    key_idx = np.where(['keyw' in k for k in dict_key_n])[0]
    auth_idx = np.where(['auth' in k for k in dict_key_n])[0]

    dict_keys_used = []

    if len(name_idx) > 0:
        name = row[dict_keys[name_idx[0]]]
        dict_keys_used.append(name)
    else:
        name = ''
    if len(desc_idx) > 0:
        desc = row[dict_keys[desc_idx[0]]]
        dict_keys_used.append(desc)
    else:
        desc = ''
    if len(key_idx) > 0:
        key = row[dict_keys[key_idx[0]]]
        dict_keys_used.append(key)
    else:
        key = ''
        
    if len(auth_idx) > 0:
        aut = row[dict_keys[auth_idx[0]]]
        dict_keys_used.append(aut)
    else:
        aut = ''
        
    other = {k: row[k] for k in dict_keys if k not in dict_keys_used}
    return [name, desc, key, aut, other]
    
    
def explode_code_and_lang(df):
    new_df = pd.DataFrame()

    print(f'Shape before exploding scripts: {df.shape}')

    for index, row in tqdm(df.iterrows()):
        if row['multiple_scripts']==True:
          for i, script in enumerate(row['code_script']):
              row['main_script'] = script
              row['main_type_script'] = row['type_script'][i]
              new_df = new_df.append(row)
        else:
          new_df = new_df.append(row)

    new_df['main_script'] = new_df['main_script'].fillna(new_df['code_script'])
    new_df['main_type_script'] = new_df['main_type_script'].fillna(new_df['type_script'])

    new_df = new_df.reset_index(drop=True)
    print(f'Shape after exploding scripts: {new_df.shape}')
    return new_df
