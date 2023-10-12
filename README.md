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
bash bash_files/run_mask_train_stage2.sh 0.3 0.3 0.3 0.7 49
```
At this stage, you can set the modality-specific sparsity, i.e., set different modality modules with different sparsity. For example, 0.3, 0.3 and 0.3 represent the compression ratio of language, vision and fusion models, respectively. 0.7 represents the zero rate (1 - compression ratio) of the whole model. 49 represents the random seeds.


Note that "FTmodel_type" and "Masker_type" in script "run_mask_train_stage2.sh" represnet the loaded model type (trained in stage1) and the training methods of mask train (stage2). By setting these two hyperparameters, models in the paper such as lmh-lpf, bce-lmh, and lmh-lmh can be obtained. There is a prerequisite here that the corresponding model must be trained in stage1. For example, to obtain lmh-lpf, it is necessary to train and obtain model lxmert(lmh) in stage1 firstly, and then use this settings (FTmodel_type="lmh", Masker_type="lpf") to run the stage2 and obtain the final model "lmh-lpf".

## For VisualBERT on VQA-CP v2
```
bash bash_files/run_mask_train_stage2_visualBert.sh 0.7 5e-5 49
```
Because visualBERT does not divide into modules according to modality, it adopts uniform sparsity (represented by zero rate, e.g., 0.7). 5e-5 represents the learning rate. 49 represents the random seeds.

## For LXMERT on VQA-VS
```
bash bash_files/run_mask_train_stage2_VQAvs.sh 0.3 0.3 0.3 0.7 49
```



# stage3 (further fine-tuning the pruned models w/ or w/o debiasing methods)
```
bash bash_files/run_vqa_stage3.sh
```
In Stage3, you need to load the model checkpoint saved in Stage1, masker checkpoint and classifierr checkpoint saved in Stage2 firstly, and then obtain the pruned model which is used to further finetune. You can set the training methods of Stahe3 by "FT_type" in script "run_vqa_stage3.sh". 




