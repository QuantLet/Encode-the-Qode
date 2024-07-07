import json
import os
import sys

import pandas as pd
from tqdm import tqdm

tqdm.pandas()
import matplotlib.pyplot as plt
import nltk
import numpy as np
import seaborn as sns
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoModelForSeq2SeqLM,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    ProgressCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    SummarizationPipeline,
    TrainerCallback,
)

nltk.download('punkt')
import sys

import evaluate

sys.path.append('../src')

import preprocessing_utils


def batch_tokenize_preprocess(batch,
                                tokenizer,
                                max_input_length,
                                max_output_length,
                                flan=False):
    

    source = batch["input_sequence"]
    target = batch["output_sequence"]

    if flan: 
        source = ["summarize code: " + snippet for snippet in source]

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
    
def generate_summary(test_samples, model, tokenizer, encoder_max_length, decoder_max_length):
    inputs = tokenizer(
        test_samples["input_sequence"],
        padding="max_length",
        truncation=True,
        max_length=encoder_max_length,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=decoder_max_length)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs, output_str

def scs_analyze(analysis_name: str,
                        model_name: str, 
                        train_data_path: str, 
                        val_data_path:   str,
                        train_data_name: str,
                        val_data_name: str,
                        encoder_max_length: int, 
                        decoder_max_length: int,
                        random_state: int, 
                        learning_rate: float=5e-5,
                        epochs: int=4, 
                        train_batch: int=4, 
                        eval_batch: int=4,
                        warmup_steps: int=500, 
                        weight_decay: float=0.1,
                        logging_stes: int=100,
                        save_total_lim: int=3,
                        label_smooting: float = 0.1,
                        predict_generate: bool=True,
                        eval_columns_list: list=[
                                "eval_loss",
                                "eval_rouge1",
                                "eval_rouge2",
                                "eval_rougeL",
                                "eval_rougeLsum",
                                "eval_bleu",
                                "eval_gen_len",
                            ],
                        save_strategy='no',
                        evaluation_strategy='no',
                        load_best_model_at_end=True,
                        evaluate_only=False,
                        report_to=None,
                        freeze=False,
            **kwargs): 
                        
    # CREATE ANALYSIS FOLDER
    ANALYSIS_FOLDER=f'reports/analysis_report_{analysis_name}'
    print(analysis_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    if model_name=='CodeT5':
        model_name="Salesforce/codet5-base-multi-sum"
        #model_name="Salesforce/codet5-base-codexglue-sum-python"
        
    elif model_name=='CodeTrans':
        model_name="SEBIS/code_trans_t5_base_source_code_summarization_python_multitask"  

    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, skip_special_tokens=False)
    
    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for i, m in enumerate(model.decoder.block):        
            #Only un-freeze the last n transformer blocks in the decoder
            if i+1 > 12 - 4:
                for parameter in m.parameters():
                    parameter.requires_grad = True

    model.to(device)
    print(device)
    
    train_dataset = load_dataset("json",
                            data_files=train_data_name,
                            field="data",
                            data_dir=train_data_path)
    

    test_dataset = load_dataset("json",
                                data_files=val_data_name,
                                field="data",
                                data_dir=val_data_path)

                                
    train_data_txt = train_dataset['train']
        
    validation_data_txt = test_dataset['train']

    if "flan-t5-base" in model_name:
        flan=True
    else:
        flan=False

    train_data = train_data_txt.map(
        lambda batch: batch_tokenize_preprocess(
            batch, 
            tokenizer=tokenizer,
            max_input_length=encoder_max_length,
            max_output_length=decoder_max_length,
            flan=flan
        ),
        batch_size=8,
        batched=True,
        remove_columns=train_data_txt.column_names,
    )
    
    validation_data = validation_data_txt.map(
        lambda batch: batch_tokenize_preprocess(
            batch, 
            tokenizer=tokenizer,
            max_input_length=encoder_max_length,
            max_output_length=decoder_max_length,
            flan=flan
        ),
        batched=True,
        remove_columns=validation_data_txt.column_names,
    )
    
    # SUBSAMPLE FOR GENERATION BEFORE TUNING
    test_samples = validation_data_txt.select(range(20))
    summaries_before_tuning = generate_summary(test_samples, 
                                                model, 
                                                tokenizer, 
                                                encoder_max_length,
                                                decoder_max_length)[1]

    
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{ANALYSIS_FOLDER}/results",
        num_train_epochs=epochs,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=train_batch,
        per_device_eval_batch_size=eval_batch,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        label_smoothing_factor=label_smooting,
        predict_with_generate=predict_generate,
        logging_dir=f"{ANALYSIS_FOLDER}/logs",
        logging_steps=logging_stes,
        save_total_limit=save_total_lim,
        report_to=report_to,
        save_strategy=save_strategy,
        logging_strategy='epoch',
        evaluation_strategy=evaluation_strategy,
        load_best_model_at_end=load_best_model_at_end
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    compute_metrics = compute_metric_with_params(tokenizer)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=validation_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # ZERO - SHOT
    results_zero_shot = trainer.evaluate()
    results_zero_shot_df = pd.DataFrame(data=results_zero_shot, index=[0])[eval_columns_list]
    results_zero_shot_df.loc[0, :] = results_zero_shot_df.loc[0, :].apply(lambda x: round(x, 3))
    print(results_zero_shot_df)
    
    results_zero_shot_df.to_csv(f'{ANALYSIS_FOLDER}/results_zero_shot.csv', index=False)
    
    if evaluate_only:
        with open(f'{ANALYSIS_FOLDER}/results.txt', "w") as results_file:
        
            for i, description in enumerate(test_samples["output_sequence"]):
                results_file.write('_'*10)
                results_file.write(f'Original: {description}')
                results_file.write(f'Summaries: {summaries_before_tuning[i]}')
        return 'Finished'
        
    
    # TRAINING
    trainer.train()
    
    # FINE-TUNING
    results_fine_tune = trainer.evaluate()
    results_fine_tune_df = pd.DataFrame(data=results_fine_tune, index=[0])[eval_columns_list]
    results_fine_tune_df.loc[0, :] = results_fine_tune_df.loc[0, :].apply(lambda x: round(x, 3))
    print(results_fine_tune_df)
    
    results_fine_tune_df.to_csv(f'{ANALYSIS_FOLDER}/results_fine_tune.csv', index=False)
    
    summaries_after_tuning = generate_summary(test_samples, 
                                                model,
                                                tokenizer,
                                                encoder_max_length,
                                                decoder_max_length)[1]
    
    for i, description in enumerate(test_samples["output_sequence"]):
        print('_'*10)
        print(f'Original: {description}')
        
        print('\n')
        print(f'Summary before Tuning: {summaries_before_tuning[i]}')
        print('\n')
        print(f'Summary after Tuning: {summaries_after_tuning[i]}')
        print('\n')
        print('_'*10)
        print('\n'*2)

    # CREATE REPORT
    with open(f'{ANALYSIS_FOLDER}/results.txt', "w") as results_file:
        
        for i, description in enumerate(test_samples["output_sequence"]):
            results_file.write('_'*10)
            results_file.write(f'Original: {description.encode("utf-8")}')
            results_file.write(f'Summary before Tuning: {summaries_before_tuning[i]}')
            results_file.write(f'Summary after Tuning: {summaries_after_tuning[i]}')
            results_file.write('_'*10)
            results_file.write('\n')
    
    
    # STORE PARAMS
    with open(f'{ANALYSIS_FOLDER}/config.json', "w") as params_file:
        config_params = {'analysis_name': analysis_name, 
                            'model_name': model_name, 
                            'train_data_path': train_data_path, 
                            'val_data_path':   val_data_path,
                            'train_data_name': train_data_name,
                            'val_data_name': val_data_name,
                            'encoder_max_length': encoder_max_length, 
                            'decoder_max_length': decoder_max_length,
                            'random_state': random_state, 
                            'learning_rate': learning_rate,
                            'epochs': epochs, 
                            'train_batch': train_batch, 
                            'eval_batch': eval_batch,
                            'warmup_steps': warmup_steps, 
                            'weight_decay': weight_decay,
                            'logging_stes': logging_stes,
                            'save_total_lim': save_total_lim,
                            'label_smooting': label_smooting,
                            'predict_generate': predict_generate,
                            'eval_columns_list': eval_columns_list,
                            'save_strategy' : save_strategy,
                            }
        json.dump(config_params, params_file)

        
    return trainer                 

def bootstrap_inference(analysis_name: str,
                            model_name: str, 
                            train_data_path: str, 
                            val_data_path:   str,
                            train_data_name: str,
                            val_data_names_list: list,
                            encoder_max_length: int, 
                            decoder_max_length: int,
                            random_state: int, 
                            learning_rate: float=5e-5,
                            epochs: int=4, 
                            train_batch: int=4, 
                            eval_batch: int=4,
                            warmup_steps: int=500, 
                            weight_decay: float=0.1,
                            logging_stes: int=100,
                            save_total_lim: int=3,
                            label_smooting: float = 0.1,
                            predict_generate: bool=True,
                            eval_columns_list: list=[
                                "eval_loss",
                                "eval_rouge1",
                                "eval_rouge2",
                                "eval_rougeL",
                                "eval_rougeLsum",
                                "eval_bleu",
                                "eval_gen_len",
                            ],
                            save_strategy='no',
                            load_best_model_at_end=True,
                            evaluate_only=False,
                        **kwargs): 
                        
    # CREATE ANALYSIS FOLDER
    ANALYSIS_FOLDER=f'reports/analysis_report_{analysis_name}'
    os.mkdir(ANALYSIS_FOLDER)
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
    
    train_dataset = load_dataset("json",
                            data_files=train_data_name,
                            field="data",
                            data_dir=train_data_path)

                                
    train_data_txt = train_dataset['train']
        
    
    
    train_data = train_data_txt.map(
        lambda batch: batch_tokenize_preprocess(
            batch, 
            tokenizer=tokenizer,
            max_input_length=encoder_max_length,
            max_output_length=decoder_max_length
        ),
        batch_size=8,
        batched=True,
        remove_columns=train_data_txt.column_names,
    )
    
    for i, val_data_name in tqdm(enumerate(val_data_names_list)):
    
        test_dataset = load_dataset("json",
                                    data_files=val_data_name,
                                    field="data",
                                    data_dir=val_data_path)
                                    
        validation_data_txt = test_dataset['train']
        
        validation_data = validation_data_txt.map(
            lambda batch: batch_tokenize_preprocess(
                batch, 
                tokenizer=tokenizer,
                max_input_length=encoder_max_length,
                max_output_length=decoder_max_length
            ),
            batched=True,
            remove_columns=validation_data_txt.column_names,
        )
    
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=f"{ANALYSIS_FOLDER}/results",
            num_train_epochs=epochs,
            do_train=True,
            do_eval=True,
            per_device_train_batch_size=train_batch,
            per_device_eval_batch_size=eval_batch,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            label_smoothing_factor=label_smooting,
            predict_with_generate=predict_generate,
            logging_dir=f"{ANALYSIS_FOLDER}/logs",
            logging_steps=logging_stes,
            save_total_limit=save_total_lim,
            report_to=None,
            save_strategy=save_strategy,
            load_best_model_at_end=load_best_model_at_end
        )
        
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        
        compute_metrics = compute_metric_with_params(tokenizer)
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_data,
            eval_dataset=validation_data,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        
        # ZERO - SHOT
        if i == 0:
            results_zero_shot = trainer.evaluate()
            results_zero_shot_df = pd.DataFrame(data=results_zero_shot, index=[0])[eval_columns_list]
            results_zero_shot_df.loc[0, :] = results_zero_shot_df.loc[0, :].apply(lambda x: round(x, 3))
            print(results_zero_shot_df)
        else: 
            results = trainer.evaluate()
            results_df = pd.DataFrame(data=results, index=[0])[eval_columns_list]
            results_df.iloc[0, :] = results_df.iloc[0, :].apply(lambda x: round(x, 3))
            
            results_zero_shot_df = pd.concat([results_zero_shot_df, results_df], axis=0)
            
    results_zero_shot_df.loc['mean', :] = results_zero_shot_df.mean(axis=0)
    results_zero_shot_df.loc['mean', :] = results_zero_shot_df.loc['mean', :].apply(lambda x: round(x, 3))
    
    results_zero_shot_df.loc['std', :]  = results_zero_shot_df.std(axis=0)
    results_zero_shot_df.loc['std', :] = results_zero_shot_df.loc['std', :].apply(lambda x: round(x, 3))
    
    results_zero_shot_df.to_csv(f'{ANALYSIS_FOLDER}/results_bootstrap.csv', index=True)
    
    
def parse_logs(trainer):
    log_history = trainer.state.log_history
    train_log = pd.DataFrame(columns=log_history[0].keys())
    eval_log = pd.DataFrame(columns=log_history[1].keys())
    for log in log_history:
        if "loss" in log:
            train_log = pd.concat(
                [train_log, pd.DataFrame.from_dict(log, orient="index").T], axis=0
            )
        elif "eval_loss" in log:
            eval_log = pd.concat(
                [eval_log, pd.DataFrame.from_dict(log, orient="index").T], axis=0
            )

    logs = train_log.merge(
        eval_log,
        how="inner",
        left_on=["epoch", "step"],
        right_on=["epoch", "step"],
    )
    return logs[
        [
            "epoch",
            "loss",
            "step",
            "eval_loss",
            "eval_rouge1",
            "eval_rouge2",
            "eval_rougeL",
            "eval_rougeLsum",
            "eval_gen_len",
            "eval_bleu",
            "eval_brevity_penalty",
            "eval_length_ratio",
            "eval_translation_length",
            "eval_reference_length",
        ]
    ]

def create_name(analysis_config):
    name = analysis_config["model_name"]
    if "checkpoint" in name:
        name = name.split("/")[-1]
    mode = analysis_config["MODE"]
    date = analysis_config["DATE"]
    if analysis_config["val_data_name"].startswith("val"):
        sample = "val"
    else:
        sample = "test"
    return f"{name}_{mode}_{sample}_{date}"