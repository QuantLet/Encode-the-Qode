"""Helper functions to preprocess code snippets and metainfo files"""

from Levenshtein import distance
import os
import json
import re
import torch
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()


def check_if_import(code_snippet, language='py'):
    
    """Check if code_snippet is part of import section"""
    is_import = False

    if language == 'py':
        if code_snippet.startswith(('import', 'from', 'pip', '!pip')):
            is_import = True
    elif language == 'r':
        if code_snippet.startswith(('library', 'install', 'source', )):
            is_import = True
    elif language == 'm':
        if code_snippet.startswith(('load', 'import')):
            is_import = True
        
    return is_import

def load_script(code_script_path: str) -> list:

    """Load the code script line by line"""

    with open(os.path.join(code_script_path), 'rb') as f:
        code = f.readlines()
    
    new_code = []
    for line in code: 
        try: 
            new_code.append(line.decode())
        except Exception as e:
            print(f"Could not laod the script because of {e} ")

    code = new_code    
    code = [item.replace('\n', '') for item in code if item.endswith('\n')]
    code = [item for item in code if len(item)>0]
    code = [f'{item}\n' for item in code]
    return code

def load_ipynb(code_script_path: str) -> list:

    """Load ipynb file and return a list of code snippets"""

    with open(code_script_path, 'rb') as f:
        data = json.load(f, strict=False)
    
    data = pd.json_normalize(data, record_path='cells')

    data = data[data['cell_type'].isin(['code', 'markdown'])]
    data = data[data['source'].notna()]
    data = data[data['source'].apply(lambda x: len(x) > 0)]
    
    data = [item for sublist in data.source.values for item in sublist]
    data = [item.replace('\n', '') for item in data if item.endswith('\n')]
    data = [item for item in data if len(item)>0]

    data = [f'{item}\n' for item in data]
    return data

def load_code(code_script_path:str, language:str='py') -> list:
    
    """Load code from file and return a list of code snippets"""
    if language == 'py':
        code = load_script(code_script_path)
    elif language == 'r':
        code = load_script(code_script_path)
    elif language == 'm':
        code = load_script(code_script_path)
    elif language == 'ipynb':
        code = load_ipynb(code_script_path)
    else:
        raise ValueError('Language not supported')
    
    return code


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch``

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def parse_meta_str(row:str) -> list:
    """ Parses metainfo file when loaded as string"""
    row = row.replace('NW: ', '')
    other = {}
    field_list = row.split("\n")
    for field in field_list:
        if ":" not in field:
            continue
        field_name, field_value = field.split(":")[0], field.split(":")[1]
        field_name = field_name.lower()
        field_value = field_value.strip()

        if "name" in field_name:
            name = field_value
        elif "desc" in field_name:
            desc = field_value
        elif "keyw" in field_name:
            key = field_value
        elif "auth" in field_name:
            aut = field_value
        else:
            other[field_name] = field_value
    return [name, desc, key, aut, other]
    
def parse_meta_dict(row:dict) -> list:
    """ Parses metainfo file when loaded as dictionary"""
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

def parse_meta(row:pd.Series) -> list: 
    """ Parses the row in pandas dataframe. Should  """
    
    row = row['metainfo_file']
    if row=='empty':
        metainfo_list = ['','','','', '']
        
    if isinstance(row, dict):
        metainfo_list = parse_meta_dict(row)
    elif isinstance(row, str):
        metainfo_list = parse_meta_str(row)
    return metainfo_list

def explode_code_and_lang(df:pd.DataFrame) -> pd.DataFrame:
    """ Creates a long pandas by turing the list of code snippets to separate rows"""

    new_df = pd.DataFrame()

    print(f'Shape before exploding scripts: {df.shape}')

    for index, row in tqdm(df.iterrows()):
        if row['multiple_scripts']:
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
    
def add_docstring_comment_tags_py(string):
    result = string.replace('\r', '')
    s_com = re.compile(r"(#*)(.*)\n")
    s_m = re.compile(r'("""|\'\'\')(.*?)\1', re.DOTALL)

    result = re.sub(s_com, r"<COMMENT S> \2 <COMMENT E>\n", result, re.DOTALL)
    result = re.sub(s_m, r'<DOCSTR START>\2<DOCSTR END>\n', result, re.DOTALL)
    return result

def add_docstring_comment_tags_r(string):
    result = string.replace('\r', '')
    s_com = re.compile(r"(#*)(.*)\n")
    s_m = re.compile(r"#'\n(.*?)\n#'", re.DOTALL)

    result = re.sub(s_com, r"<COMMENT S> \2 <COMMENT E>\n", result, re.DOTALL)
    result = re.sub(s_m, r'<DOCSTR START>\1<DOCSTR END>\n', result, re.DOTALL)
    return result

def add_docstring_comment_tags_matlab(string):
    result = string.replace('\r', '')
    s_com = re.compile(r"(%*)(.*)")
    s_m = re.compile(r"%\{\n(.*?)\n%\}", re.DOTALL)

    result = re.sub(s_com, r"<COMMENT S> \2 <COMMENT E>\n", result, re.DOTALL)
    result = re.sub(s_m, r'<DOCSTR START>\1<DOCSTR END>\n', result, re.DOTALL)
    return result

def add_docstring_comment_tags(string, lang):
    if lang=='py':
        result = add_docstring_comment_tags_py(string)
    elif lang=='m':
        result = add_docstring_comment_tags_matlab(string)
    elif lang=='r':
        result = add_docstring_comment_tags_r(string)
    return result

# remove duplicate lines
def remove_dup_lines(row):
    cleaned_up = []
    codes_list = row.split('\n')
    for cl in codes_list:
        if cl in cleaned_up:
            continue
        else:
            cleaned_up.append(cl)

    return '\n'.join(cleaned_up)

def remove_too_similar_line(row, inf_gain=0.4):
    code_splitted = row.split('\n')
    cleaned_up = []
    for i, code_line in enumerate(code_splitted):
        if i==0:
            cleaned_up.append(code_line)
        else:
            levenshtein_distance = distance(code_line, cleaned_up[-1])
            try:
                if levenshtein_distance / len(cleaned_up[-1])>=inf_gain:
                    cleaned_up.append(code_line)
            except ZeroDivisionError:
                cleaned_up.append(code_line)
    return '\n'.join(cleaned_up)

def remove_too_similar_token(row, inf_gain=0.4):
    code_splitted = row.split('\n')
    cleaned_up = []
    for i, code_line in enumerate(code_splitted):
        tokenized = code_line.split()
        new_line = []
        for j, token in enumerate(tokenized):
            if j==0:
                new_line.append(token)
            else:
                levenshtein_distance = distance(token, new_line[-1])
                if levenshtein_distance / len(new_line[-1])>=inf_gain:
                    new_line.append(token)
        new_line = ' '.join(new_line)
        cleaned_up.append(new_line)
    return '\n'.join(cleaned_up)

def cut_300(row):
    tokenized = row.split()
    return ' '.join(tokenized[:2500])
    
def greedy_clean(code_snippet):
    code_snippet = re.sub('\W+', ' ', code_snippet).strip()
    cleaned_up = [word for word in code_snippet.split() if len(word)>2]
    return ' '.join(cleaned_up)

def df_metainfo_parse(df,
    prepare_script=False,
    remove_other=False,
    remove_empty=True ):
    
    if remove_empty:
        df = df[df.metainfo_file!='empty']
    print(df.shape)
    
    COLUMNS = ['Quantlet', 'Description', 'Keywords', 'Authors', 'Other']
    

    if 'Keywords' not in df.columns:
        
        meta_info = pd.DataFrame(columns=COLUMNS)
        meta_info[COLUMNS] = df.apply(
            lambda x: parse_meta(x),
                axis='columns',
                result_type='expand'
            )

        for col in meta_info.columns:
            meta_info[col] = meta_info[col].astype(str)

        df = pd.concat([df, meta_info], axis=1)

        del df['metainfo_file']
        if remove_other:
            del df['Other']
        del df['script_name_no_ext']
        
    if prepare_script:
        df['code_script'] = df['code_script'].apply(lambda x: [line for line in x if len(line)>0])
        df['code_script'] = df['code_script'].apply(lambda x: ' '.join(x))
        
        df['scr_n'] = df['code_script'].apply(len)
        df['description_len'] = df['Description'].apply(len)
        df['description_n_words'] = df['Description'].apply(lambda x: len(x.split()))
        df = df.loc[df.Description != "",:]
        df = df.reset_index(drop=True)
        
        # ADD REPO INFORMATION
        df['repo'] = df['folder_name'].str.split('QuantLet/', expand=True)[1].str.split('/', expand=True)[0]
    print(df.shape)  
    return df

def clean_up(df): 
    
    # EXTEND THE SNIPPETS 
    df['code_script'] = df['code_script'].progress_apply(extend_tokens)

    # REMOVE CODE LINE DUPLICATES
    df['code_script'] = df['code_script'].progress_apply(remove_dup_lines)

    # REMOVE TOO SIMILAR LINES
    # we want to get as much information
    df['code_script'] = df['code_script'].progress_apply(remove_too_similar_line)

    # REMOVE TOO SIMILAR TOKENS
    df['code_script'] = df['code_script'].progress_apply(remove_too_similar_token)
    df['code_len'] = df['code_script'].progress_apply(len)

    df = df.reset_index(drop=True)
    df = df.drop(list(df[df['code_len']==0].index)).reset_index(drop=True)
    return df

def combine_url(row):
    path_ending = row["folder_name"].split(row.repo + "/")
    if len(path_ending) > 1:
        path_ending = path_ending[1]
    else:
        path_ending = path_ending[0]
    url = (
        "https://github.com/QuantLet/"
        + row["repo"]
        + "/blob/master/"
        + path_ending
        + "/"
        + row["script_name"]
    )
    return url

def save_datasets(full_train, train, val, test, DATE, RS, variable, reduced=False, save_df=False):
    if save_df:
        full_train.to_csv(
        f"../../data/preprocessed/Quantlet/{DATE}/full_train_df_{DATE}_sample0.csv",
        index=False,
        )
        train.to_csv(
            f"../../data/preprocessed/Quantlet/{DATE}/train_df_{DATE}_sample0.csv", index=False
        )
        val.to_csv(
            f"../../data/preprocessed/Quantlet/{DATE}/val_df_{DATE}_sample0.csv", index=False
        )
        test.to_csv(
            f"../../data/preprocessed/Quantlet/{DATE}/test_df_{DATE}_sample0.csv", index=False
        )

    if not reduced:
        print(train.shape)
        print(train["type_script"].value_counts(normalize=True))
        print(val.shape)
        print(val["type_script"].value_counts(normalize=True))
        print(test.shape)
        print(test["type_script"].value_counts(normalize=True))

        print(train.shape)
        print(train["type_script"].value_counts(normalize=False))
        print(val.shape)
        print(val["type_script"].value_counts(normalize=False))
        print(test.shape)
        print(test["type_script"].value_counts(normalize=False))
    
    
    
    for MODE in ["no_context", "author", "repo"]: #, 
        
        if not os.path.isdir(f"../../data/preprocessed/Quantlet/{DATE}/{MODE}"):
            os.mkdir(f"../../data/preprocessed/Quantlet/{DATE}/{MODE}")

        # FIX NA
        test.loc[test["Quantlet"].isna(), "Quantlet"] = "XFGexp_rtn_SRM_2d_DOENST RUN"
        train["Authors"] = train["Authors"].fillna("Unknown")
        val["Authors"] = val["Authors"].fillna("Unknown")
        test["Authors"] = test["Authors"].fillna("Unknown")

        if MODE == "repo":
            train.loc[:, variable] = (
                "# repo: " + train["repo"] + "\n " + train[variable]
            )
            val.loc[:, variable] = (
                "# repo: " + val["repo"] + "\n " + val[variable]
            )
            test.loc[:, variable] = (
                "# repo: " + test["repo"] + "\n " + test[variable]
            )

        elif MODE == "author":
            train.loc[:, variable] = (
                "# author: " + train["Authors"] + "\n " + train[variable]
            )
            val.loc[:, variable] = (
                "# author: " + val["Authors"] + "\n " + val[variable]
            )
            test.loc[:, variable] = (
                "# author: " + test["Authors"] + "\n " + test[variable]
            )

        train_dataset_json = {
            "version": "3.0",
            "data": [
                {
                    "input_sequence": train[variable].iloc[i],
                    "output_sequence": train["Description"].iloc[i],
                }
                for i in range(train.shape[0])
            ],
        }
        val_dataset_json = {
            "version": "3.0",
            "data": [
                {
                    "input_sequence": val[variable].iloc[i],
                    "output_sequence": val["Description"].iloc[i],
                }
                for i in range(val.shape[0])
            ],
        }

        full_train_dataset_json = {
            "version": "3.0",
            "data": [
                {
                    "input_sequence": full_train[variable].iloc[i],
                    "output_sequence": full_train["Description"].iloc[i],
                }
                for i in range(full_train.shape[0])
            ],
        }

        test_dataset_json = {
            "version": "3.0",
            "data": [
                {
                    "input_sequence": test[variable].iloc[i],
                    "output_sequence": test["Description"].iloc[i],
                }
                for i in range(test.shape[0])
            ],
        }

        with open(
            f"../../data/preprocessed/Quantlet/{DATE}/{MODE}/full_train_dataset_{DATE}_sample0.json",
            "w",
        ) as f:
            json.dump(full_train_dataset_json, f)

        with open(
            f"../../data/preprocessed/Quantlet/{DATE}/{MODE}/train_dataset_{DATE}_sample0.json",
            "w",
        ) as f:
            json.dump(train_dataset_json, f)

        with open(
            f"../../data/preprocessed/Quantlet/{DATE}/{MODE}/val_dataset_{DATE}_sample0.json",
            "w",
        ) as f:
            json.dump(val_dataset_json, f)

        with open(
            f"../../data/preprocessed/Quantlet/{DATE}/{MODE}/test_dataset_{DATE}_sample0.json",
            "w",
        ) as f:
            json.dump(test_dataset_json, f)

# Chunk code snippets
def chunk_code(code_snippet: str, chunk_size:int) -> list:
    words = code_snippet.split(' ')
    chunks = []
    current_chunk = []
    for word in words:
        if len(current_chunk) + 1 <= chunk_size:
            current_chunk.append(word)
        else:
            chunks.append(current_chunk)
            current_chunk = [word]
    if current_chunk:
        chunks.append(current_chunk)
    return list(range(len(chunks))), [' '.join(chunk) for chunk in chunks]

def camel_case_split(word):
    return re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', word)).split()

def snake_case_split(text): 
    return text.replace('_', ' ')

def extend_tokens(code_snippet): 
    code_snippet = code_snippet.replace('\n', ' \n ')
    code_snippet = snake_case_split(code_snippet)
    cleaned_cs = []
    for word in code_snippet.split(' '):
        if word == '\n': 
            cleaned_cs.append(word)
        else:
            cleaned_cs.extend(camel_case_split(word))
    code_snippet = ' '.join(cleaned_cs)   
    code_snippet = code_snippet.replace(' \n ', '\n') 
    return code_snippet