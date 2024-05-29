# CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
#     --task mquake \
#     --base_model_name gemma-7b-base \
#     --apply_chat_template False \
#     --ft_method lofit \
#     --lofit_component A \
#     --use_topk_heads 160 \
#     --lr 5e-3 \
#     --train_batch 8 \
#     --num_epoch 5 \
#     --output_dir /data/users/fcyin/finetune_checkpoints/mquake/gemma-7b-base_mquake_Aonly_seed12450\
#     --run_mode debug \
#     --output_file_name /data/users/fcyin/finetune_outputs/gemma-7b-base_mquake_Aonly_seed12450\
#     --applied_module attention \
#     --save_strategy no \
#     --l1_lambda 5e-4 \
#     --eval_batch 32 \
#     --seed 42;
CUDA_VISIBLE_DEVICES=0 python lofit_trainer.py \
    --task mquake \
    --base_model_name gemma-7b-base \
    --apply_chat_template False \
    --ft_method lofit \
    --lofit_component v \
    --use_topk_heads 16 \
    --lofit_heads top_heads/gemma-7b-base_mquake_Aonly_top160heads_42.npy\
    --lr 8e-3 \
    --train_batch 8 \
    --num_epoch 5 \
    --output_dir /data/users/fcyin/finetune_checkpoints/mquake/gemma-7b-base_mquake_Aonly_seed12450\
    --run_mode debug \
    --output_file_name /data/users/fcyin/finetune_outputs/gemma-7b-base_mquake_Aonly_seed12450\
    --applied_module attention \
    --save_strategy no \
    --l1_lambda 0 \
    --eval_batch 32 \
    --seed 42;