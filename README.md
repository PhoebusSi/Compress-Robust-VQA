# Compress-Robust-VQA
Here is the implementation of our EMNLP-2023 [Compressing And Debiasing Vision-Language Pre-Trained Models for Visual Question Answering](https://arxiv.org/abs/2210.14558). 
![image](https://github.com/PhoebusSi/Compress-Robust-VQA/blob/main/compRobustVQA.jpg)

# Stage1 (training full models w/ or w/o debiasing methods) 
```
bash bash_files/run_vqa_stage1.sh
```
Taking the training of lxmert on VQA-CP as an example. If you want to switch models or datasets, please adjust the script "run_vqa_stage1.py" yourself.

# Stage2 (pruning models w/ or w/o debiasing methods)

## For LXMERT on VQA-CP v2
```
```
## For VisualBERT on VQA-CP v2
```
```
## For LXMERT on VQA-VS
```
```
## For VisualBERT on VQA-VS
```
```


# stage3 (further fine-tuning the pruned models w/ or w/o debiasing methods)
```
```


