#!/bin/bash
# Usage: bash llama.sh <constitution> [model_id]
# model_id defaults to unsloth/Llama-3.1-8B-Instruct
# Note: --pretrain points to the distilled (merged) model, not the base

source $HOME/OpenCharacterTraining/.env
wandb login $WANDB_TOKEN

CONSTITUTION=$1
MODEL=${2:-"unsloth/Llama-3.1-8B-Instruct"}

cd $HOME

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
    --save_path $HOME/loras/llama-introspection/$CONSTITUTION \
    --eval_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 2 \
    --train_batch_size 32 \
    --zero_stage 2 \
    --seed 123456 \
    --bf16 \
    --learning_rate 5e-5 \
    --lr_warmup_ratio 0.1 \
    --max_norm 1.0 \
    --adam_betas 0.9 0.98 \
    --max_epochs 1 \
    --pretrain $HOME/models/distilled/$MODEL-$CONSTITUTION \
    --dataset $HOME/OpenCharacterTraining/data/sft_data/$MODEL/$CONSTITUTION.jsonl \
    --input_key messages \
    --apply_chat_template \
    --max_len 3072 \
    --use_wandb True \
    --wandb_project personas-llama-introspection \
    --wandb_run_name $CONSTITUTION \
    --lora_rank 64 \
    --lora_alpha 128
EOF

deepspeed \
    --module $training_commands

if [ $? -ne 0 ]; then
    echo "error: deepspeed failed"
    exit 1
fi

# remove wandb folder
rm -rf $HOME/wandb
