import os
import json
import pandas as pd
from datasets import Dataset,DatasetDict, load_dataset
import random
class TQA:
    def __init__(self,iti_split_dir,fold_num=0,data_gen_seed=42,prompt_template=None):
        self.iti_split_dir = iti_split_dir
        self.fold_num = fold_num
        self.data_gen_seed = data_gen_seed
        self.datasets = {'train':{},'val':{},'test':{}}
        
        if prompt_template is None:
            ## Default prompt template
            self.prompt_template = "Q: {question} A:"
        else:
            self.prompt_template = prompt_template
    ### Load TruthfulQA data from local directory
    def load_data(self,train_size=0):
        for split in self.datasets:
            data_dir = os.path.join(self.iti_split_dir,f'fold_{self.fold_num}_{split}_seed_{self.data_gen_seed}.csv')
            raw_data = self.tqa_formatter_csv_to_hf(json.loads(pd.read_csv(open(data_dir,'r')).to_json(orient='records')))
            ## Only sample part of the training data if train_size !=0
            if train_size!=0 and split=='train':
                raw_data = random.sample(raw_data,train_size)
            if split=='train' or split=='val':
                dpo_dict = self.format_tqa_prompts(raw_data,positive_only=False, preference=True)
                dpo_dict = Dataset.from_dict(dpo_dict)
            else:
                prompts,labels = self.format_tqa_prompts(raw_data,positive_only=False,preference=False)
            self.datasets[split]['preference_data'] = dpo_dict
            self.datasets[split]['hf']= Dataset.from_pandas(pd.DataFrame(data=self.datasets[split]))
            self.datasets[split]['data_dir'] = data_dir
            self.datasets[split]['raw_data'] = raw_data
        return self.datasets
    ### Convert TruthfulQA data in CSV to Huggingface dataset format
    def tqa_formatter_csv_to_hf(self,csv_dataset):
        hf_data = csv_dataset.copy()
        for i in range(len(hf_data)):
            entry = hf_data[i]
            hf_data[i]['question'] = entry['Question']
            hf_data[i]['mc1_targets'] = {'choices': [entry['Best Answer']] + entry['Incorrect Answers'].split('; '), 'labels': [1] + [0] * len(entry['Incorrect Answers'].split('; ')) }
            hf_data[i]['mc2_targets'] = {'choices': [entry['Best Answer']] + entry['Incorrect Answers'].split('; '), 'labels': [1]+ [0] * len(entry['Incorrect Answers'].split('; ')) }
        return hf_data
    ### Convert TruthfulQA data to preference data format for DPO
    def format_tqa_prompts(self,hf_data,key='mc2_targets',positive_only=False,preference=True,prefix=''):
        prompts = []
        labels = []
        entry_dict = {'prompt': [],'chosen':[], 'rejected': []}
        for entry in hf_data:
            prompt = self.prompt_template.format(question=entry['question'])
            entry_chosen = []
            entry_rejected = []
            for i in range(len(entry[key]['choices'])):
                label = entry[key]['labels'][i]
                prefix=prefix
                ### If prefereces == True, Format data as preference data and store them in separate lists
                if (not positive_only) and preference:
                    ### Add positive examples to the chosen examples
                    if label ==1:
                        entry_chosen.append(entry[key]['choices'][i])
                    ### Add negative examples to the rejected examples
                    else:
                        entry_rejected.append(entry[key]['choices'][i])
                ### If prefereces == False, do not format data as preference data; instead, store all choices in the same list
                if not preference:
                    choice = entry[key]['choices'][i]
                    prompt = self.prompt_template.format(question=entry['question'])
                    prompt = f"{prompt} {prefix} {choice}"
                    prompts.append(prompt)
                    labels.append(label)
            if preference:
                if len(entry_chosen)!=len(entry_rejected):
                    entry_chosen = entry_chosen[:min(len(entry_rejected),len(entry_chosen))]
                    entry_rejected = entry_rejected[:len(entry_chosen)]
                prompt = [prompt for _ in range(len(entry_chosen))]
                entry_dict['prompt'].extend(prompt)
                entry_dict['chosen'].extend(entry_chosen)
                entry_dict['rejected'].extend(entry_rejected)

        if not preference:
            return prompts,labels
        else:
            return entry_dict
class MQUAKE:
    def __init__(self,split_dir,prompt_template=None,few_shot_template=None,chat_template=False,model_name=None):
        self.split_dir = split_dir
        self.datasets = {'train':{},'val':{},'test':{}}
        
        if prompt_template is None:
            ### Default prompt template
            self.prompt_template = "Q: Imagine that {edited_prompt}. {question} A:"
        else:
            self.prompt_template = prompt_template
        if chat_template:
            self.prompt_template = f"<s>[INST] {self.prompt_template}[/INST]\n"
        self.few_shot_template = few_shot_template
        self.chat_template = chat_template
        self.model_name = model_name
    def load_data(self,train_size=0):
        for split in self.datasets:
            data_dir = os.path.join(self.split_dir,f'{split}.jsonl')
            raw_data = []
            with open(data_dir,'r') as f:
                for line in f.readlines():
                    raw_data.append(json.loads(line))
            ### Only sample part of the training data if train_size !=0
            if train_size!=0 and split=='train':
                raw_data = random.sample(raw_data,train_size)
            formatted_data = self.format_prompt(raw_data)
            self.datasets[split] = formatted_data
        return self.datasets
    def format_prompt(self,data):
        formatted_data = []
        for i in range(len(data)):
            pmeta = data[i]['requested_rewrite'][0]
            true_prompt = pmeta['prompt'].format(pmeta['subject']) + ' ' + pmeta['target_true']['str']
            edited_prompt = pmeta['prompt'].format(pmeta['subject']) + ' ' + pmeta['target_new']['str']
            questions = data[i]['questions']
            new_ans = [data[i]['new_answer']] + data[i]['new_answer_alias']
            concat_prompts = [self.prompt_template.format(edited_prompt = edited_prompt,question=q) for q in questions]
            if self.few_shot_template is not None:
                concat_prompts = [self.few_shot_template + '\n' + q for q in concat_prompts]
            ### Ignore the paraphrases, only use the first prompt + the first correct answer
            if 'gemma' in self.model_name:
                text = concat_prompts[0] + new_ans[0]
            else:
                ### If using llama models, append </s> for better performance
                text = concat_prompts[0] + new_ans[0] + '</s>'
            formatted_data.append({
                'prompt':concat_prompts[0],
                'labels':new_ans,
               'text': text
                })
        formatted_data = Dataset.from_pandas(pd.DataFrame(data=formatted_data))
        return formatted_data
class CLUTRR:
    def __init__(self,split_dir,prompt_template=None,few_shot_template=None,chat_template=False,model_name=None):
        self.split_dir = split_dir
        self.datasets = {'train':{},'val':{},'test':{}}
        
        if prompt_template is None:
            ### Default prompt template
            self.prompt_template = "Read the following story about a family. \"{}\" Assume the relations described in the story are all true. Based on relations between the fictional characters in the story (assume the relations are all true) and your commonsense knowledge about family relationship, how is {} related to {}? Answer: {} is {} 's"
        else:
            self.prompt_template = prompt_template
        if chat_template:
            self.prompt_template=f"<s>[INST] {self.prompt_template}[/INST]"    
        self.few_shot_template = few_shot_template
        self.model_name = model_name
    def load_data(self,train_size=0):
        for split in self.datasets:
            data_dir = os.path.join(self.split_dir,f'{split}.json')
            data_json = json.load(open(data_dir,'r'))
            if train_size!=0 and split=='train':
                data_json = random.sample(data_json,train_size)
            if split == 'test':
                formatted_data = self.format_clutrr_prompt(data_json,append_label=False)
            else:
                formatted_data = self.format_clutrr_prompt(data_json,append_label=True)
            self.datasets[split] = formatted_data
        return self.datasets
    def format_clutrr_prompt(self,data_json,append_label):
        data = []
        for i in range(len(data_json)):
            story = data_json[i]['clean_story']
            ## Remove the irrelevant brackets from the story in the raw dataset
            story = story.replace('[','').replace(']','')
            persons = data_json[i]['query'][1:-1].replace('\'','').split(',')
            per1 = persons[0]
            per2 = persons[1]
            gold_rel = data_json[i]['target_text']
            prompt_body = self.prompt_template.format(story,per2,per1,per2,per1)
            if append_label:
                if 'gemma' in self.model_name:
                    text = f"{prompt_body}\n{gold_rel}"
                else:
                    text = f"{prompt_body}\n{gold_rel}</s>"
                data.append({'text': text})
            else:
                text = f"{prompt_body}\n"
                data.append({'prompt': text,'target_text': gold_rel})
            
        df = pd.DataFrame(data=data)
        df = df.astype('str')
        data = Dataset.from_pandas(df)
        return data