#!/bin/sh
cd xxx/VQA/
echo "langComp$1, VisComp$2, FusComp$3, ZeroRate$4, Seed$5"
export ROOT_DIR=xxx/VQA/ 

export ZERO_RATE=$4 # # the zero-rate (1- compression ratio) of the whole model
export lang=$1 # the compression-ratio of language modules
export vis=$2 # the compression-ratio of vision modules
export fus=$3 # the compression-ratio of fusion modules
export seed=$5

export model_type=lxmert #unc-nlp/lxmert-base-uncased 
export label4save=lmh2lmh-Masker-5e-5-20epo-$lang-$vis-$fus-zerorate$ZERO_RATE-$seed 
export training_type=Masker
export output_dir=$ROOT_DIR/xxxx/$label4save/



python3 $ROOT_DIR/prune_debias_VQA.py \
    --label4save $label4save \
    --model_type $model_type \
    --masker_level modal \
    --Lang_comp $lang \
    --Vis_comp $vis \
    --Fus_comp $fus \
    --FTmodel_type lmh \
    --Masker_type lmh \
    --global_prune False \
    --model_name_or_path $ROOT_DIR/lxmert_config/ \
    --teacher_model unc-nlp/lxmert-base-uncased \
    --do_train \
    --do_eval \
    --learning_rate 5e-5 \
    --training_type $training_type \
    --evaluate_during_training \
    --output_dir $output_dir \
    --output_mask_dir $output_dir \
    --logging_dir $output_dir/log/$label4save/logging \
    --per_gpu_train_batch_size 256 \
    --per_gpu_eval_batch_size 256 \
    --num_train_epochs 20 \
    --logging_steps 100\
    --save_steps 1712 \
    --controlled_init magnitude \
    --zero_rate $ZERO_RATE \
    --seed $seed \
    --structured false \
    --use_kd false