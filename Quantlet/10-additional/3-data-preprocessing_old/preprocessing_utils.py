import os
import json
import pickle
import pandas as pd
from IPython.display import display
from torch.utils.data import Dataset
import numpy as np
import tqdm
from tqdm import tqdm
tqdm.pandas()
from Levenshtein import distance

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
        
    
def check_if_part_of_block(code_snippet, language='py'):
    
    """Check if code_snippet is part of a block of code"""
    is_block = False
    if language == 'py':
        if code_snippet.startswith(
            (
            ' ', '\t', ')', '}', ']', 'else', 'elif', 'except', 'finally'
            )
            ):
            is_block = True
    elif language == 'r':
        pass
    elif language == 'm':
        pass
    
def tokenize_block(code_list):
    
    """Tokenize a block of code"""
    
    new_list = []
    import_flag = False

    for code_snippet in code_list: 

        if check_if_part_of_block(code_snippet):
            
            new_list[-1] = new_list[-1] + code_snippet
            continue
        elif check_if_import(code_snippet):
            if import_flag:
                new_list[-1] = new_list[-1] + code_snippet
            else:
                new_list.append(code_snippet)
                import_flag = True
            continue
        else:
            new_list.append(code_snippet)
            import_flag = False

    return new_list


def load_script(code_script_path):

    with open(os.path.join(code_script_path), 'rb') as f:
        code = f.readlines()
    
    new_code = []
    for line in code: 
        try: 
            new_code.append(line.decode())
        except:
            pass
    code = new_code    
    #code = [line.decode() for line in code]
    code = [item.replace('\n', '') for item in code if item.endswith('\n')]
    code = [item for item in code if len(item)>0]
    code = [f'{item}\n' for item in code]
    #code = ''.join(code)
    return code

def load_ipynb(code_script_path):

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
    #data = ' \n'.join(data)
    #data = tokenize_block(data)
    #data = ''.join(data)
    
    return data

def load_code(code_script_path, language='py'):
    
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
            if levenshtein_distance / len(cleaned_up[-1])>=inf_gain:
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