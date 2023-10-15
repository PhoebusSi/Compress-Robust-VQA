# deepspeed==0.5.8

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
    --nnodes=1 --master_port=3227 \
    --use_env vqa_mplug.py \
    --config ./configs/full_model_debias.yaml \
    --checkpoint output/vqa_mplug_base/full_model_debias/13688.pt/mp_rank_00_model_states.pt \
    --output_dir output/vqa_mplug_base/full_model_debias_mask_debias \
    --do_two_optim \
    --add_object \
    --do_mask \
    --max_input_length 80 \
    --do_amp \
    --add_ocr \
    --deepspeed \
    --deepspeed_config ./configs/ds_config.json 
