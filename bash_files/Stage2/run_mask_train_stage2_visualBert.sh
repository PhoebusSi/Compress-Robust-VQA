#!/bin/sh
cd xxx/VQA/
echo "ZeroRate$1, LR$2, Seed$3"
export ROOT_DIR=xxx/VQA/
export ZERO_RATE=$1 #0.7


export seed=$3

export model_type=visual_bert 
export label4save=lmh2lmh-visualBert_Masker-$2-20epos-LR$2-zerorate$ZERO_RATE-$seed  
export training_type=Masker
export output_dir=xxx/CompressVQA/train_MetaL_visualBert_masker_vqa/$label4save/ 
mkdir -p $output_dir


python3 $ROOT_DIR/prune_debias_VQA_visualBERT.py \
    --label4save $label4save \
    --model_type $model_type \
    --FTmodel_type lmh \
    --Masker_type lmh \
    --global_prune False \
    --model_name_or_path uclanlp/visualbert-vqa-coco-pre \
    --do_train \
    --do_eval \
    --learning_rate $2 \
    --training_type $training_type \
    --evaluate_during_training \
    --output_dir $output_dir \
    --pred_out_dir $output_dir \
    --output_mask_dir $output_dir \
    --logging_dir $output_dir/log/$label4save/logging \
    --per_gpu_train_batch_size 256 \
    --per_gpu_eval_batch_size 256 \
    --num_train_epochs 20 \
    --logging_steps 100 \
    --save_steps 1712 \
    --controlled_init magnitude \
    --zero_rate $ZERO_RATE \
    --seed $seed \
    --structured false \
    --use_kd false