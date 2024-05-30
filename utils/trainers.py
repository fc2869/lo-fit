from datasets import Dataset,DatasetDict, load_dataset
import json,os
import pandas as pd
from transformers import AutoTokenizer, TrainingArguments, DataCollatorWithPadding, AutoModelForCausalLM,get_linear_schedule_with_warmup, Trainer,DataCollatorForLanguageModeling, BitsAndBytesConfig,AdamW,set_seed
from torch.utils.data import DataLoader
from models.modeling_llama import LlamaModel,LlamaForCausalLM
import torch 
import torch.nn as nn
from trl import DataCollatorForCompletionOnlyLM
import argparse
from trl import DPOTrainer,SFTTrainer
import numpy as np
import wandb
from accelerate import init_empty_weights
from contextlib import contextmanager, nullcontext
import random
from utils.evaluate import evaluate_mquake,evaluate_clutrr,evaluate_tqa
class CustomDPOTrainer(DPOTrainer):
    def compute_loss(self, model, inputs,return_outputs=False):
        compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        l1norm = 0
        l1_lambda=self.l1_lambda
        if l1_lambda!=0:
            for param in model.parameters():
                if param.requires_grad:
                    l1norm+=param.abs().sum()


            loss+=l1_lambda*l1norm
        return (loss, metrics) if return_outputs else loss   
    def test(self, fname=None,eval_dataset=None, ignore_keys=None, sanity_check=False, metrics=['mc'],model_name=None,**kwargs):
        if sanity_check:
            print('Sanity check...')
        self.model.eval()
        evaluate_tqa(eval_dataset,metrics=metrics,fname=fname,model=self.model,tokenizer=self.tokenizer,model_name=model_name)
        
        
class CustomSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs,return_outputs=False):
        labels = inputs['labels']

        outputs = model(**inputs)
        
        ### Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        ### We don't use .loss here since the model may return tuples instead of ModelOutput.
        cn_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        loss = cn_loss
        ### Add L1 regularization term
        l1norm = 0
        l1_lambda=self.l1_lambda
        
        for param in model.parameters():
            if param.requires_grad:
                l1norm+=param.abs().sum()
        loss+=l1_lambda*l1norm
        if return_outputs:
            return loss,outputs
        else:
            return loss
       
    def test(self, fname,task,eval_dataset=None, ignore_keys=None, sanity_check=False,model_name=None,apply_chat_template=False,**kwargs):
        if sanity_check:
            print('Sanity check...')
        self.model.eval()
        self.args.prediction_loss_only = False
        self.tokenizer.add_eos_token = False
        if not os.path.exists(fname):
            os.makedirs(fname)
        if task == 'mquake':
            generated = evaluate_mquake(eval_dataset=eval_dataset,model_name=model_name,model=self.model,tokenizer=self.tokenizer,fname=fname,apply_chat_template=apply_chat_template)
            
        elif task=='clutrr':
            generated = evaluate_clutrr(eval_dataset=eval_dataset,model_name=model_name,model=self.model,tokenizer=self.tokenizer,fname=fname,apply_chat_template=apply_chat_template)
        return generated