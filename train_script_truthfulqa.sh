## Uncomment the following to run experiments for: gemma_7b
# model_name="gemma_7b";
# task="truthfulqa";
# seed=42;
# echo $model_name;
### Experiments for fold 0 of truthulfqa two-fold cross validation
# CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
#     --task $task \
#     --base_model_name $model_name \
#     --apply_chat_template False \
#     --ft_method lofit \
#     --lofit_component A \
#     --use_topk_heads 160 \
#     --tqa_fold_num 0 \
#     --lr 5e-3 \
#     --train_batch 8 \
#     --num_epoch 5 \
#     --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_Aonly_seed${seed}"\
#     --run_mode train \
#     --output_file_name "./finetuned_outputs/${task}/${model_name}_${task}_Aonly_seed${seed}"\
#     --applied_module attention \
#     --save_strategy no \
#     --l1_lambda 5e-4 \
#     --eval_batch 32 \
#     --seed $seed;
# CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
#     --task $task  \
#     --base_model_name $model_name \
#     --apply_chat_template False \
#     --tqa_fold_num 0 \
#     --ft_method lofit \
#     --lofit_component v \
#     --use_topk_heads 16 \
#     --lofit_heads "./top_heads/${model_name}_${task}_Aonly_top160heads_${seed}.npy"\
#     --lr 2e-2 \
#     --train_batch 8 \
#     --num_epoch 5 \
#     --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_lofit_seed${seed}"\
#     --run_mode train \
#     --output_file_name "${model_name}_${task}_lofit_seed${seed}"\
#     --applied_module attention \
#     --save_strategy no \
#     --l1_lambda 0 \
#     --eval_batch 32 \
#     --seed $seed;
### Experiments for fold 1 of truthulfqa two-fold cross validation
# CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
#     --task $task \
#     --base_model_name $model_name \
#     --apply_chat_template False \
#     --ft_method lofit \
#     --lofit_component A \
#     --use_topk_heads 160 \
#     --tqa_fold_num 1 \
#     --lr 5e-3 \
#     --train_batch 8 \
#     --num_epoch 5 \
#     --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_Aonly_seed${seed}"\
#     --run_mode train \
#     --output_file_name "./finetuned_outputs/${task}/${model_name}_${task}_Aonly_seed${seed}"\
#     --applied_module attention \
#     --save_strategy no \
#     --l1_lambda 5e-4 \
#     --eval_batch 32 \
#     --seed $seed;
# CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
#     --task $task  \
#     --base_model_name $model_name \
#     --apply_chat_template False \
#     --tqa_fold_num 1 \
#     --ft_method lofit \
#     --lofit_component v \
#     --use_topk_heads 16 \
#     --lofit_heads "./top_heads/${model_name}_${task}_Aonly_top160heads_${seed}.npy"\
#     --lr 2e-2 \
#     --train_batch 8 \
#     --num_epoch 5 \
#     --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_lofit_seed${seed}"\
#     --run_mode train \
#     --output_file_name "${model_name}_${task}_lofit_seed${seed}"\
#     --applied_module attention \
#     --save_strategy no \
#     --l1_lambda 0 \
#     --eval_batch 32 \
#     --seed $seed;
## Uncomment the following to run experiments for: llama2_7B
model_name="llama2_7B";
task="truthfulqa";
seed=42;
echo $model_name;
### Experiments for fold 0 of truthulfqa two-fold cross validation
CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
    --task $task \
    --base_model_name $model_name \
    --apply_chat_template False \
    --ft_method lofit \
    --lofit_component A \
    --use_topk_heads 160 \
    --tqa_fold_num 0 \
    --lr 5e-3 \
    --train_batch 8 \
    --num_epoch 5 \
    --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_Aonly_seed${seed}"\
    --run_mode train \
    --output_file_name "./finetuned_outputs/${task}/${model_name}_${task}_Aonly_seed${seed}"\
    --applied_module attention \
    --save_strategy no \
    --l1_lambda 5e-4 \
    --eval_batch 32 \
    --seed $seed;
CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
    --task $task  \
    --base_model_name $model_name \
    --apply_chat_template False \
    --tqa_fold_num 0 \
    --ft_method lofit \
    --lofit_component v \
    --use_topk_heads 32 \
    --lofit_heads "./top_heads/${model_name}_${task}_Aonly_top160heads_${seed}.npy"\
    --lr 1e-2 \
    --train_batch 8 \
    --num_epoch 5 \
    --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_lofit_seed${seed}"\
    --run_mode train \
    --output_file_name "${model_name}_${task}_lofit_seed${seed}"\
    --applied_module attention \
    --save_strategy no \
    --l1_lambda 0 \
    --eval_batch 32 \
    --seed $seed;
### Experiments for fold 1 of truthulfqa two-fold cross validation
CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
    --task $task \
    --base_model_name $model_name \
    --apply_chat_template False \
    --ft_method lofit \
    --lofit_component A \
    --use_topk_heads 160 \
    --tqa_fold_num 1 \
    --lr 5e-3 \
    --train_batch 8 \
    --num_epoch 5 \
    --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_Aonly_seed${seed}"\
    --run_mode train \
    --output_file_name "./finetuned_outputs/${task}/${model_name}_${task}_Aonly_seed${seed}"\
    --applied_module attention \
    --save_strategy no \
    --l1_lambda 5e-4 \
    --eval_batch 32 \
    --seed $seed;
CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
    --task $task  \
    --base_model_name $model_name \
    --apply_chat_template False \
    --tqa_fold_num 1 \
    --ft_method lofit \
    --lofit_component v \
    --use_topk_heads 32 \
    --lofit_heads "./top_heads/${model_name}_${task}_Aonly_top160heads_${seed}.npy"\
    --lr 1e-2 \
    --train_batch 8 \
    --num_epoch 5 \
    --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_lofit_seed${seed}"\
    --run_mode train \
    --output_file_name "${model_name}_${task}_lofit_seed${seed}"\
    --applied_module attention \
    --save_strategy no \
    --l1_lambda 0 \
    --eval_batch 32 \
    --seed $seed;
## Uncomment the following to run experiments for: llama2_13B
# model_name="llama2_13B";
# task="truthfulqa";
# seed=42;
# echo $model_name;
### Experiments for fold 0 of truthulfqa two-fold cross validation
# CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
#     --task $task \
#     --base_model_name $model_name \
#     --apply_chat_template False \
#     --ft_method lofit \
#     --lofit_component A \
#     --tqa_fold_num 0 \
#     --use_topk_heads 160 \
#     --lr 1e-3 \
#     --train_batch 8 \
#     --num_epoch 5 \
#     --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_Aonly_seed${seed}"\
#     --run_mode train \
#     --output_file_name "./finetuned_outputs/${task}/${model_name}_${task}_Aonly_seed${seed}"\
#     --applied_module attention \
#     --save_strategy no \
#     --l1_lambda 1e-3 \
#     --eval_batch 32 \
#     --seed $seed;
# CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
#     --task $task  \
#     --base_model_name $model_name \
#     --apply_chat_template False \
#     --tqa_fold_num 0 \
#     --ft_method lofit \
#     --lofit_component v \
#     --use_topk_heads 48 \
#     --lofit_heads "./top_heads/${model_name}_${task}_Aonly_top160heads_${seed}.npy"\
#     --lr 2e-2 \
#     --train_batch 8 \
#     --num_epoch 5 \
#     --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_lofit_seed${seed}"\
#     --run_mode train \
#     --output_file_name "${model_name}_${task}_lofit_seed${seed}"\
#     --applied_module attention \
#     --save_strategy no \
#     --l1_lambda 0 \
#     --eval_batch 32 \
#     --seed $seed;
### Experiments for fold 1 of truthulfqa two-fold cross validation
# CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
#     --task $task \
#     --base_model_name $model_name \
#     --apply_chat_template False \
#     --ft_method lofit \
#     --lofit_component A \
#     --tqa_fold_num 1 \
#     --use_topk_heads 160 \
#     --lr 1e-3 \
#     --train_batch 8 \
#     --num_epoch 5 \
#     --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_Aonly_seed${seed}"\
#     --run_mode train \
#     --output_file_name "./finetuned_outputs/${task}/${model_name}_${task}_Aonly_seed${seed}"\
#     --applied_module attention \
#     --save_strategy no \
#     --l1_lambda 1e-3 \
#     --eval_batch 32 \
#     --seed $seed;
# CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
#     --task $task  \
#     --base_model_name $model_name \
#     --apply_chat_template False \
#     --tqa_fold_num 1 \
#     --ft_method lofit \
#     --lofit_component v \
#     --use_topk_heads 48 \
#     --lofit_heads "./top_heads/${model_name}_${task}_Aonly_top160heads_${seed}.npy"\
#     --lr 2e-2 \
#     --train_batch 8 \
#     --num_epoch 5 \
#     --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_lofit_seed${seed}"\
#     --run_mode train \
#     --output_file_name "${model_name}_${task}_lofit_seed${seed}"\
#     --applied_module attention \
#     --save_strategy no \
#     --l1_lambda 0 \
#     --eval_batch 32 \
#     --seed $seed;