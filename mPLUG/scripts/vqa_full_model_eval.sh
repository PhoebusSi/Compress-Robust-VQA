# deepspeed==0.5.8

CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 \
    --nnodes=1 --master_port=3225 \
    --use_env vqa_mplug.py \
    --config ./configs/full_model.yaml \
    --checkpoint output/vqa_mplug_base/full_model/13688.pt/mp_rank_00_model_states.pt \
    --output_dir output/vqa_mplug_base/full_model/test_result \
    --do_two_optim \
    --add_object \
    --evaluate \
    --max_input_length 80 \
    --do_amp \
    --add_ocr \
    --deepspeed \
    --deepspeed_config ./configs/ds_config.json 
