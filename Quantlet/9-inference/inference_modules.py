import os
import pickle
import json
import re
import sys

from tqdm import tqdm
import pandas as pd
tqdm.pandas()
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import  DataLoader
from transformers import AutoTokenizer, AutoModelWithLMHead, SummarizationPipeline
from transformers import AdamW
from datasets import load_dataset

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

import nltk
nltk.download('punkt')
import evaluate

import sys
sys.path.append('../src')

import preprocessing_utils

def batch_tokenize_preprocess(batch,
                              tokenizer,
                              max_input_length,
                              max_output_length):

    source = batch["input_sequence"]
    target = batch["output_sequence"]

    source_tokenized = tokenizer(
        source,
        padding="max_length",
        truncation=True,
        max_length=max_input_length
    )

    target_tokenized = tokenizer(
        target,
        padding="max_length",
        truncation=True,
        max_length=max_output_length
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
    preds  = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels
    
def compute_metric_with_params(tokenizer, metrics_list=['rouge', 'bleu']):
    def compute_metrics(eval_preds):
    
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
    
            prediction_lens = [
                np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
            ]
            result["gen_len"] = np.mean(prediction_lens)
            result = {k: round(v, 4) for k, v in result.items()}
            results_dict.update(result)
        return results_dict
    return compute_metrics
    
def generate_summary(test_samples, model, tokenizer, encoder_max_length):
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

def inference(analysis_name: str,
             model_name: str, 
             test_samples: list, 
             encoder_max_length: int, 
             decoder_max_length: int,
             random_state: int): 
                         
    # CREATE ANALYSIS FOLDER
    os.mkdir(f'inference_{analysis_name}')
    print(analysis_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    if model_name=='CodeT5':
        model_name="Salesforce/codet5-base-multi-sum"
        
    elif model_name=='CodeTrans':
        model_name="SEBIS/code_trans_t5_base_source_code_summarization_python_multitask"
                         
    model = AutoModelWithLMHead.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, skip_special_tokens=False)
    model.to(device)
    print(device)
    
    summaries = generate_summary(test_samples, 
                                model, 
                                tokenizer, 
                                encoder_max_length)[1]
    
    return summaries
    
    