# LoFiT: Localized Fine-tuning on LLM Representations

This paper provides code for the paper [LoFiT: Localized Fine-tuning on LLM Representations](arxiv.org). In this work, we introduce LoFiT, a two-step localized fine-tuning method for LLMs that selects a subset of attention heads and learns task-specific offset vectors to be added to the hidden representations of the targeted attention heads. We show the strong downstream performance of LoFiT on tasks involving truthfulness and reasoning, outperforming representation intervention methods (ITI and RepE) and matching strong PEFT methods (LoRA) with fewer learned parameters.

## Abstract
> Recent work in interpretability shows that large language models (LLMs) can be adapted for new tasks in a learning-free way: it is possible to intervene on LLM representations to elicit desired behaviors for alignment. For instance, adding certain bias vectors to the outputs of certain attention heads is reported to boost the truthfulness of models. In this work, we show that localized fine-tuning serves as an effective alternative to such representation intervention methods. We introduce a framework called Localized Fine-Tuning on LLM Representations LoFiT, which identifies a subset of attention heads that are most important for learning a specific task, then trains offset vectors to add to the model's hidden representations at those selected heads. LoFiT localizes to a sparse set of heads (3%) and learns the offset vectors from limited training data, comparable to the settings used for representation intervention. For truthfulness and reasoning tasks, we find that LoFiT's intervention vectors are more effective for LLM adaptation than vectors from representation intervention methods such as Inference-time Intervention. We also find that the localization step is important: selecting a task-specific set of attention heads can lead to higher performance than intervening on heads selected for a different task. Finally, for the tasks we study, LoFiT achieves comparable performance to other parameter-efficient fine-tuning methods such as LoRA, despite modifying 20x-200x fewer parameters than these methods.

## Table of Contents
1. [Installation](#installation)
2. [Data](#data)
3. [Train and Evaluate](#train-and-evaluate)
4. [How to Cite](#how-to-cite)

## Installation
We suggest using Python 3.8.10. Before building the environment, please install the appropriate PyTorch version that corresponds to the hardware configurations (especially GPUs) of your machine here: https://pytorch.org/get-started/locally/
Then, run the following.
```
# Setup virtual environmnet
python3.8 -m venv "lofit"
source lofit/bin/activate
# install requirements
pip install -r requirements.txt
# Create the directory to store the finetuned checkpoints and evaluation outputs
mkdir finetuned_checkpoints
mkdir finetuned_outputs
mkdir tqa_results
# Create the directory to store important attention heads selected by LoFiT 
mkdir top_heads
```
## Data
We use TruthfulQA, MQuAKE, and CLUTRR to evaluate LoFiT. The pre-processing of these datasets is described in Section 4 of our paper. We include the train/val/test splits we used for each dataset under ```datasets```. You can use 
## Train and Evaluate
### Setting up the models
We currently support Gemma 7B, Llama 2-7B, and Llama 2-13B as base models to fine-tune. Paths to the huggingface checkpoints of these models should be defined in ```models_map``` in ```lofit_trainer.py``` before running any training or evaluation script. All fine-tuning experiments can be run on a single 48G GPU.
### End-to-end training
We provide an end-to-end script to run head selection, bias tuning, and final evaluation of LoFiT for each dataset in one line as ```train_script_{task}.sh```. For example, to fine-tune Llama 2-7B-base on TruthfulQA with LoFiT, please run the following:
```
bash train_script_truthfulqa.sh
```
Specific hyperparameter configurations might be needed for different models and different datasets. Please refer to our paper for details.
### Evaluation
We integrate the evaluation step and the training step into the end-to-end script mentioned above. The codes to evaluate LoFiT on TruthfulQA are adapted from the codebase of [Inference-time Intervention](https://github.com/likenneth/honest_llama). Details of evaluating LoFiT on MQuAKE and CLUTRR can be found in ```utils/evaluate.py```.

## How to Cite