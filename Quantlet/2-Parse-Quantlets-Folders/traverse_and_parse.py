import pandas as pd
import yaml
import os
import pickle
from tqdm import tqdm
import sys
sys.path.append('../3-data-preprocessing')
import preprocessing_utils

def traverse_folder(path, file_types): 
    
    repos = []
    nw = 0
    code_file_counter = 0
    
    for i, (root, directories, files) in tqdm(enumerate(os.walk(path))):

            if '.git' in root:
                 continue
            #print(f'Processing {i} folder: {root}...')


            m_file = [file for file in files if file.lower()=="metainfo.txt"] 

            if len(m_file)==0:
                continue

            elif len(m_file)!=0:
                
                try:
                    with open(f'{root}/{m_file[0]}', 'r') as meta:
                        metainfofile = yaml.safe_load(meta)
                except:
                    try:
                        with open(f'{root}/{m_file[0]}', 'r') as meta:
                            metainfofile =  'NW: ' + meta.read()
                    except:
                        print(f'{root}/{m_file[0]}')

                code_files = [file for file in files if file.split('.')[-1] in file_types] 


                if len(code_files)!=0:

                    for code_i, c_file in enumerate(code_files):

                        q = {}

                        language = c_file.split('.')[-1].lower()

                        try:
                            sc_file = preprocessing_utils.load_code(f'{root}/{c_file}', language=language)
                        except:
                            sc_file = 'empty'

                        q['folder_name'] = root
                        q['metainfo_file'] = metainfofile
                        q['code_script'] = sc_file
                        q['type_script'] = language
                        q['script_name'] = c_file
                        repos.append(q)
                        code_file_counter += 1 
                        
    return {'repos' : repos, 'nw' : nw, 'counter' : code_file_counter}


def prepare_repos_df(repos: list) -> pd.DataFrame:
    repos_df = pd.DataFrame(repos)
    repos_df = repos_df[~((repos_df.code_script=='empty')&(repos_df.metainfo_file=='empty')&(repos_df.type_script=='empty'))]
    #repos_df['q_name'] = repos_df.folder_name.str[26:]
    repos_df = repos_df.drop_duplicates(['folder_name', 'script_name'])
    print(repos_df.shape)

    repos_df = repos_df[repos_df.code_script!='empty']
    repos_df = repos_df.drop_duplicates(['folder_name', 'script_name'])
    print(repos_df.shape)
    
    # REMOVE DUPLICATES IPYNB PY (that were created intially)
    repos_df['script_name_no_ext'] = repos_df.script_name.str.split('.', expand=True)[0]

    py_scripts = repos_df[repos_df.type_script=='py'].script_name_no_ext
    ipy_scripts = repos_df[repos_df.type_script=='ipynb'].script_name_no_ext

    both = list(set(ipy_scripts).intersection(set(py_scripts)))
    both = [f'{q}.py' for q in both]

    repos_df = repos_df[~repos_df.script_name.isin(both)]
    print(repos_df.shape)
    
    return repos_df