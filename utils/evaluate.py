import os
import torch
import json
from honest_llama.utils import alt_tqa_evaluate
def evaluate_mquake(eval_dataset,model_name,model,tokenizer,fname,batch_size=16,max_new_tokens=16,apply_chat_template=False):
    results_dir = os.path.join(fname,'outputs.json')
    results_json = []
    tokenizer.padding_side = 'left'
    inputs = eval_dataset['prompt']
    iterator = range(0, len(inputs), batch_size)
    generated = []
    with torch.no_grad():
        for i in iterator:
            inputs_b = inputs[i:i+batch_size]
            inputs_b = tokenizer(inputs_b, return_tensors='pt',padding=True)
            inputs_b= {k:v.to(model.device) for (k,v) in inputs_b.items()}
            outputs = model.generate(**inputs_b,max_new_tokens=max_new_tokens,do_sample=False)
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated.extend(decoded_outputs)
    corr=0
    for i in range(len(generated)):
        
        seq = generated[i]
        if apply_chat_template:
            sep = '[/INST]\n'
        else:
            sep = 'A:'
        ### Extract answer from the generated output
        ans = seq.split(sep)[-1].split('\n')[0].lower()
        entry_corr = 0
        ### Check if the answer is in the list of correct answers
        if ans in [label.lower() for label in eval_dataset['labels'][i]]:
            entry_corr=1
        corr+=entry_corr
        result = {'prompt': inputs[i],'response':seq,'pred':ans,'label':eval_dataset['labels'][i],'correct':entry_corr}
        results_json.append(result)
    print('Accuracy:',corr/len(generated))

    json.dump(results_json,open(results_dir,'w'))
    return generated
def evaluate_clutrr(eval_dataset,model_name,model,tokenizer,fname,batch_size=16,max_new_tokens=16,apply_chat_template=False):
    results_dir = os.path.join(fname,'outputs.json')
    result_json = []
    tokenizer.padding_side = 'left'
 
    inputs = eval_dataset['prompt']
    iterator = range(0, len(inputs), batch_size)
    generated = []
    with torch.no_grad():
        for i in iterator:
            inputs_b = inputs[i:i+batch_size]
            inputs_b = tokenizer(inputs_b, return_tensors='pt',padding=True)
            inputs_b= {k:v.to(model.device) for (k,v) in inputs_b.items()}
            outputs = model.generate(**inputs_b,max_new_tokens=max_new_tokens,do_sample=False)
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated.extend(decoded_outputs)
    corr=0
    for i in range(len(generated)):
        
        seq = generated[i]
        if apply_chat_template:
            sep = '[/INST]\n'
        else:
            sep = "\'s\n"
        ### Extract answer from the generated output
        ans = seq.split(sep)[1].split('.')[0].strip().lower()
        ### Check if the answer is equal to the gold label
        gold_label = eval_dataset['target_text'][i].lower()
        entry_corr= 0
        if ans == gold_label:
            entry_corr = 1
        corr+=entry_corr
        result = {'prompt': inputs[i],'response':seq,'pred':ans,'label':gold_label,'correct':entry_corr}
        result_json.append(result)
    print('Accuracy:',corr/len(generated))
    json.dump(result_json,open(results_dir,'w'))
    return generated
def evaluate_tqa(eval_dataset,fname,model,tokenizer,metrics,model_name=None,verbose=False):
    mnames = {
        'llama2-7b-base': 'llama2_7B',
        'llama2-13b-base': 'llama2_13B',
        'gemma-7b-base': 'gemma_7b',
        'llama2-7b-chat': 'llama2_chat_7B'
    }
    ### Create directories to save truthfulqa outputs
    if not os.path.exists('./tqa_results/answer_dump'):
        os.makedirs('./tqa_results/answer_dump')
    if not os.path.exists('./tqa_results/summary_dump'):
        os.makedirs('./tqa_results/summary_dump')
    curr_fold_results = alt_tqa_evaluate(
        {mnames[model_name]: model}, 
        metric_names=metrics,
        input_path=eval_dataset['data_dir'], 
        output_path=f'./tqa_results/answer_dump/{fname}.csv',
        summary_path=f'./tqa_results/summary_dump/{fname}.csv',
        device="cuda", 
        tokenizer=tokenizer,
        ### Print generated outputs
        verbose = verbose,
        ### Use the standard QA prompt for evaluation
        preset='qa'
    )
    print(curr_fold_results)