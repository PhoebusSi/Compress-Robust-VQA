#!/bin/bash
export TASK_NAME=VQA
export ROOT_DIR=xxxx/VQA/CompressAndRobust/mask_train/
export ZERO_RATE=0.7
export model_type=lxmert #unc-nlp/lxmert-base-uncased 
export output_dir=$ROOT_DIR/models/xxx-detailed-configs-for-model-saving 
mkdir -p $output_dir
export label4save=LMHlxmert 
export training_type=FTlmh #chose from ['FTonly', 'FTlmh'] 
export logging_dir=$output_dir/log/$training_type{_}$label4save/logging

python3 run_vqa_stage1.py \
  --label4save $label4save \
  --FT_type lmh \
  --model_type $model_type \
  --model_name_or_path  unc-nlp/lxmert-base-uncased \
  --task_name $TASK_NAME \
  --do_train \
  --training_type $training_type \
  --evaluate_during_training \
  --output_mask_dir $output_dir \
  --per_gpu_train_batch_size 64 \
  --per_gpu_eval_batch_size 64 \
  --logging_steps 1000 \
  --warmup_steps 34235 \
  --save_steps 6847 \
  --controlled_init magnitude \
  --structured false \
  --use_kd false \
  --learning_rate 5e-5 \
  --num_train_epochs 20 \
  --output_dir $output_dir \
  --logging_dir $logging_dir \
  --mask_dir None \
  --root_dir $ROOT_DIR \
  --zero_rate $ZERO_RATE