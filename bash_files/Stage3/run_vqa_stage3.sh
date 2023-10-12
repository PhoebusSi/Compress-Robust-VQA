#!/bin/bash
export TASK_NAME=VQA
export ROOT_DIR=xxxx/VQA/CompressAndRobust/mask_train/
export PRUN_TYPE=mag #mag or rand
export ZERO_RATE=0.5
export model_type=lxmert #unc-nlp/lxmert-base-uncased 
export base_dir=$ROOT_DIR/models/train_masker_vqa/FTedlxmert-normalMasker-0.5-6e-5-linearLR-20epo-allMask #path of masker and classifier that saved in Stage2

export mask_dir=$base_dir/mask.pt
export clf_dir=$base_dir/classifier4masker.bin
mkdir -p $output_dir
export label4save=normal2normal2normal  #FTlxmert #The first part before ‘2’ automatically determines the type of pretrained model to load: normal or LMH 
export new_LR4stage3=5e-5
export output_dir=$base_dir/$label4save-$new_LR4stage3-$ZERO_RATE
#Note: "label4save" of stage3's scripts must be in the form of "A2B2C". Because the pretrained model to load needs to be determined based on A.

export training_type=FT_trainedMask  
export logging_dir=$output_dir/log/$training_type/logging




CUDA_VISIBLE_DEVICES=3 nohup python3 $ROOT_DIR/run_vqa_stage3.py \
  --mask_dir $mask_dir \
  --clf_dir $clf_dir \
  --label4save $label4save \
  --FT_type normal \
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
  --learning_rate $new_LR4stage3 \
  --num_train_epochs 20 \
  --output_dir $output_dir \
  --logging_dir $logging_dir \
  --root_dir $ROOT_DIR \
  --prun_type $PRUN_TYPE \
  --zero_rate $ZERO_RATE  
