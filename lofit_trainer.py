from datasets import Dataset,DatasetDict, load_dataset
import json,os
import pandas as pd
from transformers import AutoTokenizer, TrainingArguments, DataCollatorWithPadding, AutoModelForCausalLM,get_linear_schedule_with_warmup, Trainer,DataCollatorForLanguageModeling, BitsAndBytesConfig,AdamW,set_seed,logging
from torch.utils.data import DataLoader
from models.modeling_llama import LlamaModel,LlamaForCausalLM
from models.modeling_gemma import GemmaForCausalLM
import torch 
import torch.nn as nn
from trl import DataCollatorForCompletionOnlyLM
import argparse
from trl import DPOTrainer
import numpy as np
import wandb
from accelerate import init_empty_weights
from contextlib import contextmanager, nullcontext
import random
from peft import LoraConfig, TaskType, get_peft_model, LoraModel, PeftModel
from utils.dataloaders import TQA,MQUAKE,CLUTRR
from utils.trainers import CustomDPOTrainer,CustomSFTTrainer

parser = argparse.ArgumentParser()
parser.add_argument('--lr',type=float,default=1e-4)
parser.add_argument('--train_batch',type=int,default=16)
parser.add_argument('--num_epoch',type=int,default=10)
parser.add_argument('--train_size',type=int,default=0)
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--output_dir', type=str,default=None)
parser.add_argument('--eval_batch',type=int,default=8)
parser.add_argument('--task', type=str,help='The task dataset to train on')
parser.add_argument('--run_mode', type=str, default='train',help='The mode to run the script: train or train_wandb. Train: train the model; train_wandb: train the model and log the results to wandb.')
parser.add_argument('--output_file_name',type=str,help='The name of the output file')
parser.add_argument('--dpo_beta', type=float, default=0.5,required=False,help='The hyperparameter of beta value for DPO')
parser.add_argument('--applied_module',type=str,default='attention',help='The modules to apply lofit; attention by default')
parser.add_argument('--applied_layers',type=str,default=None,help='The list of layers to apply lofit; None by default and it means apply lofit to all layers')
parser.add_argument('--l1_lambda', type=float, default=0, help='l1 regularization lambda for lofit',required=False)
parser.add_argument('--base_model_name',type=str,default='llama2-7b-base',help='The model base to train on',required=True)
parser.add_argument('--lofit_component',type=str,default='full',help='Choose the components to apply acfit. A: head selection step; v: bias tuning step',required=False)
parser.add_argument('--ft_method',type=str,default='lofit',help='fine-tuning method to apply',required=True)
parser.add_argument('--lofit_heads',type=str,default=None,help='Load a .npy file where the top heads from the head selection step are stored',required=False)
parser.add_argument('--hf_cache_dir',type=str,default='/data/users/fcyin/.cache',required=False,help='The cache directory for huggingface models')
parser.add_argument('--device',type=str,default='cuda',required=False,help='The device to load the model; cuda by default')
parser.add_argument('--save_strategy',type=str,default='best',required=False,help='The strategy to save the model: best: only save the best model; no: do not save the model')
parser.add_argument('--tqa_fold_num',type=int,default=0,required=False,help='The fold number to use for truthfulqa; can only be 0 or 1 for two-fold cross validation')
parser.add_argument('--apply_chat_template',default=False, type=lambda x: (str(x).lower() == 'true'),help='Using llama2 chat template in the prompt; False by default')
parser.add_argument('--use_topk_heads',type=int,help='The number of top attention heads to select; if in the head selection step, K means only save the top-k heads; if in the bias tuning step, K means only use the top-k heads from the loaded top heads to tune the biases')
args = parser.parse_args()
### Turn Wandb log on if it is in train mode
if args.run_mode == 'train_wandb':
    wandb.init(mode="online",name=args.output_dir.split("/")[-1])
else:
    wandb.init(mode="disabled")
### Load training hyperparametres
lr = float(args.lr)
train_batch_size = int(args.train_batch)
eval_batch_size = int(args.eval_batch)
sample_size = int(args.train_size)
num_epoch = int(args.num_epoch)
applied_module = args.applied_module
l1_lambda = args.l1_lambda
dpo_beta = args.dpo_beta
output_dir = args.output_dir
device = args.device
lofit_heads = args.lofit_heads
topk_heads = args.use_topk_heads
head_selection_strategy = args.head_selection
### If lofit_heads is not None, assert the heads are stored in a numpy file and load it into a numpy array
### Format of the npy file: each row is a tuple of (layer,head); heads are sorted by their importance score from the head selection step in descending order
if lofit_heads is not None:
    assert '.npy' in lofit_heads
    ### Only use the topk_heads heads
    lofit_heads = np.load(lofit_heads)[:topk_heads,:]
    ### Convert np array to list of tuples
    lofit_heads = list(zip(lofit_heads[:,0], lofit_heads[:,1]))
if args.applied_layers is not None:
    applied_layers = list(map(int,args.applied_layers.split(',')))
else:
    applied_layers = None
## Set all random seeds for reproducibility
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
set_seed(seed)

logging.set_verbosity_error()
### Maps of model names and task names
### TO-DO: Define your models here
models_map = {
    'llama2-7b-chat': '/data/shared_resources/models/llama2/hf/llama-2-7b-chat',
    'llama2-7b-base': '/data/shared_resources/models/llama2/hf/llama-2-7b',
    'llama2-13b-base': '/data/shared_resources/models/llama2/hf/llama-2-13b',
    'gemma-7b-base': '/data/shared_resources/models/gemma/gemma-7b'
}
task_map = {
    'truthfulqa': {
        'dataloader': TQA(
    iti_split_dir = './dataset/truthfulqa',
    fold_num = args.tqa_fold_num,
    data_gen_seed = 42
),
        'trainer': CustomDPOTrainer
    },
    'mquake': {
        'dataloader': MQUAKE(
            split_dir = './dataset/MQuAKE',
            chat_template = args.apply_chat_template,
            model_name = args.base_model_name
        ),
        'trainer': CustomSFTTrainer
    },
    'clutrr':{
        'dataloader': CLUTRR(
            split_dir = './dataset/clutrr',
            chat_template = args.apply_chat_template,
            model_name = args.base_model_name
        ),
        'trainer': CustomSFTTrainer
    },
}
if not args.base_model_name in models_map:
    raise ValueError(f'The base model {args.base_model_name} is not supported')
### Load tokenizers and models
model_name = models_map[args.base_model_name]
cache_dir = args.hf_cache_dir
tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir)

### Use right padding for training
tokenizer.padding_side = 'right'    
if 'gemma' in model_name:
    tokenizer.add_bos_token = True
    tokenizer.add_eos_token = True
if '13b' in model_name or 'gemma' in model_name:
    ## Use bfloat16 training for 13B models and Gemma
    torch_dtype = torch.bfloat16
    bf16 = True
else:
    torch_dtype = torch.float32
    bf16 = False
peft_config = None
if args.ft_method == 'lofit':
    if 'llama' in model_name:
        model = LlamaForCausalLM.custom_from_pretrained(model_name,
                                                device_map=device, 
                                                cache_dir=cache_dir,
                                                applied_module = applied_module,
                                                applied_layers = applied_layers,
                                                torch_dtype=torch_dtype)
    elif 'gemma' in model_name:
        model = GemmaForCausalLM.custom_from_pretrained(model_name,
                                                device_map=device, 
                                                cache_dir=cache_dir,
                                                applied_module = applied_module,
                                                applied_layers = applied_layers,
                                                torch_dtype=torch_dtype)
    else:
        raise ValueError(f'Fine-tuning method {args.ft_method} for {model_name} is not supported!')
else:
    raise ValueError(f'Fine-tuning method {args.ft_method} is not supported!')
### Define padding
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(model.config.vocab_size + 1)   

count = 0
if args.run_mode!='test':
    ### First freeze all pretrained parameters
    for param in model.parameters():
        param.requires_grad = False
    trainable_params = []
    num_params = 0
    ### Unfreeze LoFiT parameters for training
    for i in range(model.config.num_hidden_layers):
        if applied_module == 'attention':
            if args.lofit_component == 'A':
                attn_A = model.model.layers[i].self_attn.attn_A
                for j,module in enumerate(attn_A):

                    trainable_params.append(module)
                    module.requires_grad = True
                    num_params+=module.numel()
            if args.lofit_component == 'v':
                attn_v = model.model.layers[i].self_attn.attn_v
                for j,module in enumerate(attn_v):
                    if lofit_heads is None or (i,j) in lofit_heads:
                        trainable_params.append(module)
                        module.requires_grad = True
                        num_params+=module.numel()
                        count+=1
        else:
            raise ValueError(f'Fine-tuning {applied_module} is supported yet!')
    print('trainable params:',num_params)
    # optimizer = AdamW(trainable_params, lr=lr)
if args.save_strategy == 'best':
    save_strategy = 'epoch'
    load_best_model_at_end = True
    save_total_limit = 1
elif args.save_strategy == 'no':
    save_strategy = 'no'
    load_best_model_at_end = False
    save_total_limit = None
else:
    raise ValueError(f'Save strategy {args.save_strategy} is not supported')
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=lr,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    num_train_epochs=num_epoch,
    evaluation_strategy="epoch",
    save_strategy=save_strategy,
    load_best_model_at_end=load_best_model_at_end,
    save_total_limit = save_total_limit,
    report_to='wandb',
    logging_strategy='epoch',
    seed = seed,
    do_train = True,
    do_eval = True,
    bf16=bf16
)
torch.autograd.set_detect_anomaly(True)
datasets = task_map[args.task]['dataloader'].load_data(train_size=args.train_size)
for key in ['train','val','test']:
    print(f"Number of {key} samples: {len(datasets[key])}")
    # print(datasets[key])
trainer = task_map[args.task]['trainer']

if args.task == 'truthfulqa':
    trainer = trainer(
        model = model,
        ref_model = None,
        train_dataset=datasets['train']['preference_data'],
        eval_dataset = datasets['val']['preference_data'],
        tokenizer=tokenizer,
        max_prompt_length=128,
        max_target_length=128,
        args=training_args,
        ### Temperature for DPO training
        beta = args.dpo_beta,
        ### Precompute log probabilities of data of the reference model to speed up training and save GPU memory
        precompute_ref_log_probs=True
    )
elif args.task in ['mquake','clutrr']:
    if args.task == 'mquake':
        response_template_with_context = " A:"
    elif args.task == 'clutrr':
        response_template_with_context = " 's\n"
    if args.apply_chat_template:
        response_template_with_context = " [/INST]\n"
    if 'llama' in model_name:
        ### Special thing about llama tokenizer
        
        response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]
    elif 'gemma' in model_name:
        response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[1:]
    ### DataCollatorForCompletionOnlyLM is used for updating loss ONLY on the response
    ### If you want to do standard LM loss (i.e. loss update on both the prompt and the response), you don't need to use this data collator 
    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    
    trainer = trainer(
        model,
        train_dataset=datasets['train'],
        eval_dataset = datasets['val'],
        dataset_text_field = 'text',
        tokenizer=tokenizer,
        max_seq_length=400,
        data_collator = data_collator,
        args=training_args,
        peft_config = peft_config
    )
if args.run_mode!='test':
    trainer.l1_lambda = l1_lambda
    if args.ft_method == 'lofit':
        
        for i in range(model.config.num_hidden_layers):
            if applied_module == 'attention':
                if args.lofit_component == 'A':
                    attn_A = model.model.layers[i].self_attn.attn_A
                    for j,module in enumerate(attn_A):
                        ### Use miu_{A} = 0, sigma_{A} = 1e-3 as the default
                        nn.init.normal_(module,mean=0,std=1e-3)
                if args.lofit_component == 'v':
                    attn_v = model.model.layers[i].self_attn.attn_v
                    for j,module in enumerate(attn_v):
                        if lofit_heads is None or (i,j) in lofit_heads:
                            ### Use miu_{v} = 0, sigma_{v} = 1e-3 as the default
                            nn.init.normal_(module,mean=0,std=1e-3)
    trainer.train(
    )
if args.lofit_component=='A':
    ### Save the top heads after finishing learning the scalars
    num_layers = trainer.model.config.num_hidden_layers
    num_heads = trainer.model.config.num_attention_heads
    vhead = np.zeros(shape=(num_layers,num_heads))
    ahead = np.zeros(shape=(num_layers,num_heads))
    for i in range(num_layers):
        for j in range(num_heads):
            ahead[i,j] =  np.linalg.norm(trainer.model.model.layers[i].self_attn.attn_A[j].data.cpu().to(torch.float32).numpy())
    f = lambda x: (x//num_heads,x%num_heads)
    k=args.use_topk_heads if args.use_topk_heads is not None else int(0.1*num_heads * num_layers)
    print(f'Number of Attention Heads Selected: {k}')
    if args.lofit_component=='A':
        topk = np.argsort(ahead.flatten())[::-1][:k]
    tuples = f(topk)
    top_tuples = []
    for i in range(k):
        top_tuples.append((tuples[0][i],tuples[1][i]))
    ### Create a directory to store the tope heads
    if not os.path.exists(f"./top_heads"):
        os.makedirs(f"./top_heads")
    if args.lofit_component=='A':
        np.save(f"./top_heads/{args.base_model_name}_{args.task}_Aonly_top{k}heads_{args.seed}.npy",np.array(top_tuples))

if args.task == 'truthfulqa':
    trainer.test(fname=args.output_file_name,eval_dataset = datasets['test'],model_name = args.base_model_name)
elif args.task in ['mquake','clutrr']:
    trainer.test(fname=args.output_file_name,task=args.task,eval_dataset = datasets['test'],model_name = args.base_model_name)

