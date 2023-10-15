# deepspeed==0.5.8

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 \
    --nnodes=1 --master_port=3225 \
    --use_env vqa_mplug.py \
    --config ./configs/full_model.yaml \
    --checkpoint ./ckpts/mplug_base.pth \
    --output_dir output/vqa_mplug_base/full_model_debias \
    --do_two_optim \
    --add_object \
    --max_input_length 80 \
    --do_amp \
    --add_ocr \
    --deepspeed \
    --deepspeed_config ./configs/ds_config.json 
