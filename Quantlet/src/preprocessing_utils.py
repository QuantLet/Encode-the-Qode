import os
import json
import pickle
import pandas as pd
from IPython.display import display
from torch.utils.data import Dataset
import numpy as np

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

class QuantletDataset(Dataset):
    def __init__(self, parsed_Qs_file):
        with open(parsed_Qs_file, 'rb') as f:
            loaded_dataset = pickle.load(f)
        
        # make sure the index is continuous    
        self.parsed_Qs_file = loaded_dataset

        self.dw = 0

    def __len__(self):
        return len(self.parsed_Qs_file.script_name)

    def __getitem__(self, idx):
        subset = self.parsed_Qs_file.iloc[idx, :]
        code_script_path = os.path.join(subset.folder_name, subset.script_name)
        try:
            code_snippet = load_code(code_script_path, subset.type_script)
        except:
            self.dw += 1
            code_snippet = 'empty'
        return code_snippet
    
    def show_df(self):
        return display(self.parsed_Qs_file) 
    
'''class QuantletDataloader(Dataloader):
    def __init__(self, parsed_Qs_file):
        self.dataset = QuantletDataset(parsed_Qs_file)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]'''

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
    other = {k: row[k] for k in dict_keys if k not in dict_keys_used}
    return [name, desc, key, other]

def batch_tokenize_preprocess(batch,
                              tokenizer,
                              max_source_length,
                              max_target_length):

    source = batch["input_sequence"]
    target = batch["output_sequence"]

    source_tokenized = tokenizer(
        source,
        padding="max_length",
        truncation=True,
        max_length=max_source_length
    )

    target_tokenized = tokenizer(
        target,
        padding="max_length",
        truncation=True,
        max_length=max_target_length
    )

    batch = {k: v for k, v in source_tokenized.items()}

    # Ignore padding in the loss

    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in target_tokenized["input_ids"]
    ]

    return batch
    
def postprocess_text(preds, labels):

    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels
    
def compute_metrics(eval_preds, metrics_list=['rouge', 'bleu']):

    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # POST PROCESSING
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    results_dict = {}
    for m in metrics_list:
        metric = evaluate.load(m)

        if m=='bleu':
            result = metric.compute(
              predictions=decoded_preds, references=decoded_labels
           )
        elif m=='rouge':
            result = metric.compute(
                predictions=decoded_preds, references=decoded_labels, use_stemmer=True
            )
        result = {key: value for key, value in result.items() if key!='precisions'}
        #result = {key: value * 100 for key, value in result.items() if 'rouge' in key}

        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        results_dict.update(result)
    return results_dict
     
def generate_summary(test_samples, model):
    inputs = tokenizer(
        test_samples["input_sequence"],
        padding="max_length",
        truncation=True,
        max_length=encoder_max_length,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    outputs = model.generate(input_ids, attention_mask=attention_mask)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs, output_str