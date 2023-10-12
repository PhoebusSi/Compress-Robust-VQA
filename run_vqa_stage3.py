# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""

from collections import defaultdict, Counter
import dataclasses
from hg_transformers import TrainingArguments as BaseTrainingArguments
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch.nn.utils.prune as prune
from dataset_LXM import Dictionary, VQAFeatureDataset
import json

import numpy as np
from hg_transformers.data.metrics import compute_score_with_logits
import torch

import sys
sys.path.append('transformer/src/')

from hg_transformers.configuration_auto import AutoConfig
from hg_transformers.modeling_auto import AutoModelForSequenceClassification, AutoModelForMultipleChoice
from hg_transformers.tokenization_auto import AutoTokenizer
from hg_transformers.trainer_utils import EvalPrediction
from hg_transformers.data.datasets.glue import GlueDataset
from hg_transformers.data.datasets.glue import GlueDataTrainingArguments as DataTrainingArguments
from hg_transformers.hf_argparser import HfArgumentParser
#from hg_transformers.trainer import Trainer
from hg_transformers.mask_trainer_VQA import Trainer
from hg_transformers.training_args import TrainingArguments
from hg_transformers.data.processors.glue import glue_output_modes
from hg_transformers.data.processors.glue import glue_tasks_num_labels
from hg_transformers.data.metrics import glue_compute_metrics
from hg_transformers.trainer import set_seed
from hg_transformers.optimization import get_linear_schedule_with_warmup, get_constant_schedule, get_constant_schedule_with_warmup
#from optimization import AdamW
from hg_transformers.optimization import AdamW
from hg_transformers import PreTrainedTokenizer, MODEL_WITH_LM_HEAD_MAPPING, TrimCollator

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


default_params = {
        "vqa": {"max_seq_length": 14, 'per_gpu_train_batch_size': 128, 'num_train_epochs': 20, 'learning_rate': 1e-5}
    }

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def see_weight_rate(model, model_type):
    sum_list = 0
    zero_sum = 0
    for ii in range(9):
        sum_list = sum_list+float(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.query.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.query.weight_mask'] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.key.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.key.weight_mask'] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.value.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.value.weight_mask'] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.output.dense.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.output.dense.weight_mask'] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.intermediate.dense.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.intermediate.dense.weight_mask'] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.output.dense.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.output.dense.weight_mask'] == 0))

    for ii in range(5):
        sum_list = sum_list+float(model.state_dict()['%s.encoder.r_layers.'%model_type+str(ii)+'.attention.self.query.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.r_layers.'%model_type+str(ii)+'.attention.self.query.weight_mask'] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.r_layers.'%model_type+str(ii)+'.attention.self.key.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.r_layers.'%model_type+str(ii)+'.attention.self.key.weight_mask'] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.r_layers.'%model_type+str(ii)+'.attention.self.value.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.r_layers.'%model_type+str(ii)+'.attention.self.value.weight_mask'] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.r_layers.'%model_type+str(ii)+'.attention.output.dense.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.r_layers.'%model_type+str(ii)+'.attention.output.dense.weight_mask'] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.r_layers.'%model_type+str(ii)+'.intermediate.dense.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.r_layers.'%model_type+str(ii)+'.intermediate.dense.weight_mask'] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.r_layers.'%model_type+str(ii)+'.output.dense.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.r_layers.'%model_type+str(ii)+'.output.dense.weight_mask'] == 0))


    for ii in range(5):
        sum_list = sum_list+float(model.state_dict()['%s.encoder.x_layers.'%model_type+str(ii)+'.visual_attention.att.query.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.x_layers.'%model_type+str(ii)+'.visual_attention.att.query.weight_mask'] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.x_layers.'%model_type+str(ii)+'.visual_attention.att.key.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.x_layers.'%model_type+str(ii)+'.visual_attention.att.key.weight_mask'] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.x_layers.'%model_type+str(ii)+'.visual_attention.att.value.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.x_layers.'%model_type+str(ii)+'.visual_attention.att.value.weight_mask'] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.x_layers.'%model_type+str(ii)+'.visual_attention.output.dense.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.x_layers.'%model_type+str(ii)+'.visual_attention.output.dense.weight_mask'] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.x_layers.'%model_type+str(ii)+'.lang_self_att.self.query.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.x_layers.'%model_type+str(ii)+'.lang_self_att.self.query.weight_mask'] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.x_layers.'%model_type+str(ii)+'.lang_self_att.self.key.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.x_layers.'%model_type+str(ii)+'.lang_self_att.self.key.weight_mask'] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.x_layers.'%model_type+str(ii)+'.lang_self_att.self.value.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.x_layers.'%model_type+str(ii)+'.lang_self_att.self.value.weight_mask'] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.x_layers.'%model_type+str(ii)+'.lang_self_att.output.dense.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.x_layers.'%model_type+str(ii)+'.lang_self_att.output.dense.weight_mask'] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.x_layers.'%model_type+str(ii)+'.visn_self_att.self.query.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.x_layers.'%model_type+str(ii)+'.visn_self_att.self.query.weight_mask'] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.x_layers.'%model_type+str(ii)+'.visn_self_att.self.key.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.x_layers.'%model_type+str(ii)+'.visn_self_att.self.key.weight_mask'] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.x_layers.'%model_type+str(ii)+'.visn_self_att.self.value.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.x_layers.'%model_type+str(ii)+'.visn_self_att.self.value.weight_mask'] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.x_layers.'%model_type+str(ii)+'.visn_self_att.output.dense.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.x_layers.'%model_type+str(ii)+'.visn_self_att.output.dense.weight_mask'] == 0))


        sum_list = sum_list+float(model.state_dict()['%s.encoder.x_layers.'%model_type+str(ii)+'.lang_inter.dense.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.x_layers.'%model_type+str(ii)+'.lang_inter.dense.weight_mask'] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.x_layers.'%model_type+str(ii)+'.lang_output.dense.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.x_layers.'%model_type+str(ii)+'.lang_output.dense.weight_mask'] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.x_layers.'%model_type+str(ii)+'.visn_inter.dense.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.x_layers.'%model_type+str(ii)+'.visn_inter.dense.weight_mask'] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.x_layers.'%model_type+str(ii)+'.visn_output.dense.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.x_layers.'%model_type+str(ii)+'.visn_output.dense.weight_mask'] == 0))



    sum_list = sum_list+float(model.state_dict()['%s.pooler.dense.weight_mask'%model_type].nelement())
    zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.pooler.dense.weight_mask'%model_type] == 0))
    sum_list = sum_list+float(model.state_dict()['%s.embeddings.word_embeddings.weight_mask'%model_type].nelement())
    zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.embeddings.word_embeddings.weight_mask'%model_type] == 0))
    print("model.state_dict()", model.state_dict().keys())
    sum_list = sum_list+float(model.state_dict()['%s.encoder.visn_fc.visn_fc.weight_mask'%model_type].nelement())
    zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.visn_fc.visn_fc.weight_mask'%model_type] == 0))
    sum_list = sum_list+float(model.state_dict()['%s.encoder.visn_fc.box_fc.weight_mask'%model_type].nelement())
    zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.visn_fc.box_fc.weight_mask'%model_type] == 0))
    return 100*zero_sum/sum_list


def summarize_results(task, output_dir):
    lines = """import re, os
import numpy as np

task = '%s'"""%task.lower() + """
seeds = [i for i in range(1, 4)]
pattern = re.compile(r'-?\d+\.?\d*e?-?\d*?')

scores = []
for seed in seeds:
        filename = os.path.join(str(seed), 'eval_results_%s.txt'%task)
        file = open(filename, 'r')
        lines = file.readlines()
        s = float(pattern.findall(lines[-1])[0])
        print('%d: %.3f'%(seed, s))
        scores.append(s)
        file.close()
score = np.mean(scores)
std = np.std(scores)
print('Avg score: %.3f'%(score))
print('Std: %.3f'%(std))
    """
    if not os.path.exists(output_dir[:-2]+'/summarize_results.py'):
        file = open(os.path.join(output_dir[:-2], 'summarize_results.py'), 'w')
        file.write(lines)
        file.close()


def mag_pruning(model,px):

    print('Start magnitude pruning with zero rate %.2f'%px)
    modules_to_prune =[]
    for ii in range(12):
        modules_to_prune.append('encoder.layer.%d.attention.self.query'%ii)
        modules_to_prune.append('encoder.layer.%d.attention.self.key'%ii)
        modules_to_prune.append('encoder.layer.%d.attention.self.value'%ii)
        modules_to_prune.append('encoder.layer.%d.attention.output.dense'%ii)
        modules_to_prune.append('encoder.layer.%d.intermediate.dense'%ii)
        modules_to_prune.append('encoder.layer.%d.output.dense'%ii)

    modules_to_prune.append('pooler.dense')
    for name, module in model.named_modules():
        if name in modules_to_prune:
            prune.l1_unstructured(module, 'weight', amount=px)
    prune.l1_unstructured(model.embeddings.word_embeddings, 'weight', amount=px)

def pruning_model_with_mask(model, mask_dict, model_type):
    parameters_to_prune =[]
    mask_list = []
    suffix = '.weight_mask' if '_mask' in list(mask_dict.keys())[0] else '.weight'
    for ii in range(9):
        parameters_to_prune.append(model.encoder.layer[ii].attention.self.query)
        mask_list.append(mask_dict['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.query%s'%suffix])
        parameters_to_prune.append(model.encoder.layer[ii].attention.self.key)
        mask_list.append(mask_dict['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.key%s'%suffix])
        parameters_to_prune.append(model.encoder.layer[ii].attention.self.value)
        mask_list.append(mask_dict['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.value%s'%suffix])
        parameters_to_prune.append(model.encoder.layer[ii].attention.output.dense)
        mask_list.append(mask_dict['%s.encoder.layer.'%model_type+str(ii)+'.attention.output.dense%s'%suffix])
        parameters_to_prune.append(model.encoder.layer[ii].intermediate.dense)
        mask_list.append(mask_dict['%s.encoder.layer.'%model_type+str(ii)+'.intermediate.dense%s'%suffix])
        parameters_to_prune.append(model.encoder.layer[ii].output.dense)
        mask_list.append(mask_dict['%s.encoder.layer.'%model_type+str(ii)+'.output.dense%s'%suffix])
    for ii in range(5):
        parameters_to_prune.append(model.encoder.r_layers[ii].attention.self.query)
        mask_list.append(mask_dict['%s.encoder.r_layers.'%model_type+str(ii)+'.attention.self.query%s'%suffix])
        parameters_to_prune.append(model.encoder.r_layers[ii].attention.self.key)
        mask_list.append(mask_dict['%s.encoder.r_layers.'%model_type+str(ii)+'.attention.self.key%s'%suffix])
        parameters_to_prune.append(model.encoder.r_layers[ii].attention.self.value)
        mask_list.append(mask_dict['%s.encoder.r_layers.'%model_type+str(ii)+'.attention.self.value%s'%suffix])
        parameters_to_prune.append(model.encoder.r_layers[ii].attention.output.dense)
        mask_list.append(mask_dict['%s.encoder.r_layers.'%model_type+str(ii)+'.attention.output.dense%s'%suffix])
        parameters_to_prune.append(model.encoder.r_layers[ii].intermediate.dense)
        mask_list.append(mask_dict['%s.encoder.r_layers.'%model_type+str(ii)+'.intermediate.dense%s'%suffix])
        parameters_to_prune.append(model.encoder.r_layers[ii].output.dense)
        mask_list.append(mask_dict['%s.encoder.r_layers.'%model_type+str(ii)+'.output.dense%s'%suffix])
    for ii in range(5):
        parameters_to_prune.append(model.encoder.x_layers[ii].visual_attention.att.query)
        mask_list.append(mask_dict['%s.encoder.x_layers.'%model_type+str(ii)+'.visual_attention.att.query%s'%suffix])
        parameters_to_prune.append(model.encoder.x_layers[ii].visual_attention.att.key)
        mask_list.append(mask_dict['%s.encoder.x_layers.'%model_type+str(ii)+'.visual_attention.att.key%s'%suffix])
        parameters_to_prune.append(model.encoder.x_layers[ii].visual_attention.att.value)
        mask_list.append(mask_dict['%s.encoder.x_layers.'%model_type+str(ii)+'.visual_attention.att.value%s'%suffix])
        parameters_to_prune.append(model.encoder.x_layers[ii].visual_attention.output.dense)
        mask_list.append(mask_dict['%s.encoder.x_layers.'%model_type+str(ii)+'.visual_attention.output.dense%s'%suffix])
        parameters_to_prune.append(model.encoder.x_layers[ii].lang_self_att.self.query)
        mask_list.append(mask_dict['%s.encoder.x_layers.'%model_type+str(ii)+'.lang_self_att.self.query%s'%suffix])
        parameters_to_prune.append(model.encoder.x_layers[ii].lang_self_att.self.key)
        mask_list.append(mask_dict['%s.encoder.x_layers.'%model_type+str(ii)+'.lang_self_att.self.key%s'%suffix])
        parameters_to_prune.append(model.encoder.x_layers[ii].lang_self_att.self.value)
        mask_list.append(mask_dict['%s.encoder.x_layers.'%model_type+str(ii)+'.lang_self_att.self.value%s'%suffix])
        parameters_to_prune.append(model.encoder.x_layers[ii].lang_self_att.output.dense)
        mask_list.append(mask_dict['%s.encoder.x_layers.'%model_type+str(ii)+'.lang_self_att.output.dense%s'%suffix])
        parameters_to_prune.append(model.encoder.x_layers[ii].visn_self_att.self.query)
        mask_list.append(mask_dict['%s.encoder.x_layers.'%model_type+str(ii)+'.visn_self_att.self.query%s'%suffix])
        parameters_to_prune.append(model.encoder.x_layers[ii].visn_self_att.self.key)
        mask_list.append(mask_dict['%s.encoder.x_layers.'%model_type+str(ii)+'.visn_self_att.self.key%s'%suffix])
        parameters_to_prune.append(model.encoder.x_layers[ii].visn_self_att.self.value)
        mask_list.append(mask_dict['%s.encoder.x_layers.'%model_type+str(ii)+'.visn_self_att.self.value%s'%suffix])
        parameters_to_prune.append(model.encoder.x_layers[ii].visn_self_att.output.dense)
        mask_list.append(mask_dict['%s.encoder.x_layers.'%model_type+str(ii)+'.visn_self_att.output.dense%s'%suffix])
        parameters_to_prune.append(model.encoder.x_layers[ii].lang_inter.dense)
        mask_list.append(mask_dict['%s.encoder.x_layers.'%model_type+str(ii)+'.lang_inter.dense%s'%suffix])
        parameters_to_prune.append(model.encoder.x_layers[ii].lang_output.dense)
        mask_list.append(mask_dict['%s.encoder.x_layers.'%model_type+str(ii)+'.lang_output.dense%s'%suffix])     
        parameters_to_prune.append(model.encoder.x_layers[ii].visn_inter.dense)
        mask_list.append(mask_dict['%s.encoder.x_layers.'%model_type+str(ii)+'.visn_inter.dense%s'%suffix])
        parameters_to_prune.append(model.encoder.x_layers[ii].visn_output.dense)
        mask_list.append(mask_dict['%s.encoder.x_layers.'%model_type+str(ii)+'.visn_output.dense%s'%suffix])     
    parameters_to_prune.append(model.pooler.dense)
    parameters_to_prune.append(model.embeddings.word_embeddings)
    mask_list.append(mask_dict['%s.pooler.dense%s'%(model_type, suffix )])
    mask_list.append(mask_dict['%s.embeddings.word_embeddings%s'%(model_type, suffix)])
    parameters_to_prune.append(model.encoder.visn_fc.visn_fc)
    parameters_to_prune.append(model.encoder.visn_fc.box_fc)
    mask_list.append(mask_dict['%s.encoder.visn_fc.visn_fc%s'%(model_type, suffix)])
    mask_list.append(mask_dict['%s.encoder.visn_fc.box_fc%s'%(model_type, suffix)])

    for ii in range(len(parameters_to_prune)):
        prune.CustomFromMask.apply(parameters_to_prune[ii], 'weight', mask=mask_list[ii])






def prune_with_mask(model, mask_dir, component_type):
    """
    component_type = 'head' or 'ffn'
    """
    logger.info("Loading mask from %s"%mask_dir)
    mask = torch.from_numpy(np.load(mask_dir))
    to_prune = {}
    for layer in range(len(mask)):
        to_mask = [h[0] for h in (1 - mask[layer].long()).nonzero().tolist()]
        to_prune[layer] = to_mask
    assert sum(len(h) for h in to_prune.values()) == (1 - mask.long()).sum().item()

    logger.info("%s zero rate:%.3f"%(component_type, (mask==0).view(-1).sum().div(float(mask.numel()))))
    if component_type=='head':
        logger.info(f"Pruning heads {to_prune}")
        model.prune_heads(to_prune)
    elif component_type=='ffn':
        model.prune_ffns(to_prune)


@dataclass
class TrainingArguments(BaseTrainingArguments):
    """
    This is a subclass of transformers.TrainingArguments
    """
    use_kd: str2bool = field(
        default=False, metadata={"help": "Whether to use KD."}
    )
    # 'Masker' is the necessary setting for stage2. 
    training_type: Optional[str] = field(
        default=None, metadata={"help": "FTlmh, FTlpf, FTrubi, FTonly, FT_trainedMask, FT_randMask."}
    )
    #the training methods (w/ or w/o debiasing methods) during mask train
    FT_type: Optional[str] = field(
        default=None, metadata={"help": "normal, lmh, lpf, rubi"}
    )
    #This option determines the path of the loaded model
    label4save: Optional[str] = field(
        default=None, metadata={"help": "FT model method + masker methoddebiasing + continueFT/or not, e.g., normal+lmh+continueFT"}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    teacher_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "The teacher model (for KD) checkpoint for weights initialization."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    zero_rate: Optional[float] = field(
        default=0., metadata={"help": "The percentate of 0 in model weights."}
    )
    # mgnitu
    prun_type: Optional[str] = field(
        default='mag', metadata={"help": "To use magnitude pruning or random pruning. mag or rand"}
    )
    threshold: Optional[float] = field(
        default=1e-2, metadata={"help": "The threshold for masking."}
    )
    init_scale: Optional[float] = field(
        default=2e-2, metadata={"help": "For initialization the real-value mask matrices."}
    )
    mask_classifier: str2bool = field(
        default=False, metadata={"help": "Whether to mask classifier weights."}
    )
    mask_biases: str2bool = field(
        default=False, metadata={"help": "Whether to mask biases."}
    )
    force_masking: Optional[str] = field(
        default='bert', metadata={"help": "?", "choices": ["all", "bert", "classifier"]}
    )
    controlled_init: Optional[str] = field(
        default=None, 
        metadata={"help": "To use magnitude pruning or random pruning. mag or rand",
                "choices": ["magnitude", "uniform", "magnitude_and_uniform", "double_uniform"]}
    )
    structured_masking: Optional[str] = field(
        default=None, metadata={"help": "Whether to perform structured masking."}
    )
    structured_masking_types: Optional[str] = field(
        default=None, metadata={"help": "The type of structured masking."}
    )
    name_of_masker: Optional[str] = field(
        default='MaskedLinear1', metadata={"help": "To type of masker to use."}
    )
    layers_to_mask: Optional[str] = field(
        default='0,1,2,3,4,5,6,7,8,9,10,11', metadata={"help": "The layers to mask."}
    )
    root_dir: Optional[str] = field(
        default=None, metadata={"help": "The root directory."}
    )
    load_head_mask_dir: Optional[str] = field(
        default=None, metadata={"help": "The directory of head mask for initialization."}
    )
    load_ffn_mask_dir: Optional[str] = field(
        default=None, metadata={"help": "The directory of FFN mask for initialization."}
    )
    output_mask_dir: Optional[str] = field(
        default=None, metadata={"help": "The directory to save mask."}
    )
    masking_scheduler_conf: Optional[str] = field(
        default='lambdas_lr=0,sparsity_warmup=automated_gradual_sparsity,sparsity_warmup_interval_epoch=0.1,init_epoch=0,final_epoch=1', 
        metadata={"help": "Configurations for making scheduler."}
    )
    structured: str2bool = field(
        default=True, metadata={"help": "Whether to use structured pruning."}
    )
    train_head_mask: str2bool = field(
        default=True, metadata={"help": "Whether to train head mask."}
    )
    train_ffn_mask: str2bool = field(
        default=True, metadata={"help": "Whether to train FFN mask."}
    )
    freeze_mlm_head: str2bool = field(
        default=False, metadata={"help": "Whether to freeze mlm head parameters."}
    )
    save_mlm_head: str2bool = field(
        default=True, metadata={"help": "Whether to save mlm head parameters."}
    )
    mask_dir: Optional[str] = field(
        default=None, metadata={"help": "The usntructured mask directory."}
    )
    clf_dir: Optional[str] = field(
        default=None, metadata={"help": "The classifier trained with usntructured mask directory."}
    )
    pred_model_dir: Optional[str] = field(
        default=None, metadata={"help": "The model which is used for prediction and output json files."}
    )
    pred_out_dir: Optional[str] = field(
        default=None, metadata={"help": "The dir where the json files will save."}
    )
    #the path of pretrained models obtaned in Stage1 (training with BCE: FTlxmert; training with LMH: lmhFTlxmert; training with LPF: lpfFTlxmert; training with rubi: rubiFTlxmert)
    FTlxmert_dir: Optional[str] = field(
        default="xxxx/VQA/CompressAndRobust/mask_train/models/pretrained_model/LXMERT_48", metadata={"help": "The model fine-tuned normmally directory."}
    )
    lmhFTlxmert_dir: Optional[str] = field(
        default="xxxx/VQA/CompressAndRobust/mask_train/models/pretrained_model/LXMLMH_63", metadata={"help": "The model fine-tuned with LMH  directory."}
    )
    lpfFTlxmert_dir: Optional[str] = field(
        default="xxxx/VQA/CompressVQA/pretrained_model/LXMLPF", metadata={"help": "The model fine-tuned with LPF  directory."}
    )
    rubiFTlxmert_dir: Optional[str] = field(
        default="xxxx/VQA/CompressVQA/pretrained_model/LXMRUBI", metadata={"help": "The model fine-tuned with rubi  directory."}
    )    
    
    ffn_mask_dir: Optional[str] = field(
        default=None, metadata={"help": "The ffn mask directory."}
    )
    head_mask_dir: Optional[str] = field(
        default=None, metadata={"help": "The head mask directory."}
    )
    mask_seed: Optional[int] = field(
        default=1, metadata={"help": "The seed for random masking."}
    )
    prune_ffn: str2bool = field(
        default=False, metadata={"help": "Whether to prune FFNs."}
    )
    prune_head: str2bool = field(
        default=False, metadata={"help": "Whether to prune attention heads."}
    )
    structured: str2bool = field(
        default=False, metadata={"help": "Whether to perform structured masking."}
    )
    struc_prun_type: Optional[str] = field(
            default="one_step", metadata={"help": "One step or iterative pruning.", "choices":["one_step", "iterative", "random"]}
    )



@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """



    dataroot: Optional[str] = field(
        default='xxxx/CompressVQA/vqacpv2/', metadata={"help": "The input text training data file (a text file)."}
    )
    img_root: Optional[str] = field(
        default='xxxx/CompressVQA/coco/', metadata={"help": "The input img training data file (a text file)."}
    )

    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    ratio : float = field(
        default=1, metadata={"help": "Ratio of training set used"}
    )

    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": "Ratio of length of a span of masked tokens to surrounding context length for permutation language modeling."
        },
    )
    max_span_length: int = field(
        default=5, metadata={"help": "Maximum length of a span of masked tokens for permutation language modeling."}
    )

    ans_num: int = field(
        default=2274, metadata={"help": "Maximum size of label space."}
    )



    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def get_dataset(args: ModelArguments, tokenizer: PreTrainedTokenizer, evaluate=False):
    #file_path = args.eval_data_file if evaluate else args.train_data_file
    mode = 'test' if evaluate else 'train'
    #tokenizer = tokenizer
    """
    if not 'wiki' in file_path:
        args.line_by_line = True
    if args.line_by_line:
        return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)
    else:
        return TextDataset(
            tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, overwrite_cache=args.overwrite_cache
        )
    """
    return VQAFeatureDataset(mode, args.dataroot, args.img_root, ratio=1.0,  tokenizer=tokenizer, adaptive=False) 


def init_optimizer(model, training_args, num_train_data):
    params = [
            {
                "params": [value],
                "name": key,
                "weight_decay": training_args.weight_decay,
                "param_size": value.size(),
                "nelement": value.nelement(),
                "lr": training_args.learning_rate,
            }
            for key, value in model.named_parameters()
            if value.requires_grad
        ]
    optimizer = torch.optim.Adam(params, lr= training_args.learning_rate, betas=(0.9, 0.999), eps= training_args.adam_epsilon) 
    num_training_steps = int(int(num_train_data/(training_args.n_gpu*\
                     training_args.per_gpu_train_batch_size)+1)*training_args.num_train_epochs)
    #scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=training_args.warmup_steps)
    #scheduler = get_constant_schedule(optimizer)
    scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=num_training_steps)
    opt = (optimizer,scheduler)
    return opt


def main():
    def get_answer(p, dataloader):
        _m, idx = p.max(0)
        return dataloader.dataset.label2ans[idx.item()]

    def make_json(logits, qIds, dataloader):
        assert  logits.size(0)==len(qIds)
 
        results = []
        for i in range(logits.size(0)):
            result = {}
            result['question_id'] = qIds[i].item()
            result['answer'] = get_answer(logits[i], dataloader)
            results.append(result)
        return results
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    #parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed) 


    if training_args.do_train and training_args.do_predict:
        print("do_train", training_args.do_train)
        print("do_predict", training_args.do_predict)
        print("You must not choose both do_train and do_eval at the same time, choose one of them please!")
        assert 1==2


    print("loading config")
    config = AutoConfig.from_pretrained("xxxx/VQA/lxmert_config/") 
    if training_args.do_train:

        if training_args.training_type in ['FTonly', 'FTlmh', 'FTlpf', 'FTrubi']: 
            model = AutoModelForMultipleChoice.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
            )
        elif training_args.training_type in ['FT_trainedMask','FT_randMask']:
            if training_args.label4save[:6] == 'normal':
                model = torch.load(model_args.FTlxmert_dir+'/FTlxmert_FTonly.bin').cpu()
                if isinstance(model,torch.nn.DataParallel):
                    model = model.module

            elif training_args.label4save[:3] == 'lmh':
                model = torch.load(model_args.lmhFTlxmert_dir+'/LMHlxmert_FTlmh_only.bin').cpu()
                if isinstance(model,torch.nn.DataParallel):
                    model = model.module
            elif training_args.label4save[:3] == 'lpf':
                model = torch.load(model_args.lpfFTlxmert_dir+'/FTlpf_NoCompress_FTlpf_only.bin').cpu()
                if isinstance(model,torch.nn.DataParallel):
                    model = model.module
            elif training_args.label4save[:4] == 'rubi':
                model = torch.load(model_args.rubiFTlxmert_dir+'/FTrubi_NoCompress_FTrubi_only.bin').cpu()
                if isinstance(model,torch.nn.DataParallel):
                    model = model.module  
            else:
                print("Plz make sure that the label4save is in A2B2C form, we use A to determine the dir to load the stage1 checkpoint")
                assert 1==2

    print("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("/opt/tiger/VQA/lxmert_config/")#/mnt/sda/erqing/VQA/CompressAndRobust/mask_train/lxmert_config")
 

    if training_args.do_predict:
        print("If you wanna do predict, please set the pred_model_dir(model) and the pred_out_dir(ans_json).")
        assert model_args.pred_model_dir != None
        assert model_args.pred_out_dir != None
        model = torch.load(model_args.pred_model_dir).cpu()
        #model = model.to(training_args.device)
        if isinstance(model,torch.nn.DataParallel):
            model = model.module    



    # Get datasets
    train_dataset = get_dataset(data_args, tokenizer=tokenizer) #if training_args.do_train else None
    eval_dataset = get_dataset(data_args, tokenizer=tokenizer, evaluate=True)
    test_dataset = get_dataset(data_args, tokenizer=tokenizer, evaluate=True)
    data_collator = TrimCollator()
    
    #for LMH
    answer_voc_size = data_args.ans_num #train_dset.num_ans_candidates

    # Compute the bias:
    # The bias here is just the expected score for each answer/question type

    # question_type -> answer -> total score
    question_type_to_probs = defaultdict(Counter)
    # question_type -> num_occurances
    question_type_to_count = Counter()
    for ex in train_dataset.entries:
        ans = ex["answer"]
        q_type = ans["question_type"]
        question_type_to_count[q_type] += 1
        if ans["labels"] is not None:
            for label, score in zip(ans["labels"], ans["scores"]):
                question_type_to_probs[q_type][label] += score

    question_type_to_prob_array = {}
    for q_type, count in question_type_to_count.items():
        prob_array = np.zeros(answer_voc_size, np.float32)
        for label, total_score in question_type_to_probs[q_type].items():
            prob_array[label] += total_score
        prob_array /= count
        question_type_to_prob_array[q_type] = prob_array

    # Now add a `bias` field to each example
    for ds in [train_dataset, eval_dataset]:
        for ex in ds.entries:
            q_type = ex["answer"]["question_type"]
            ex["bias"] = question_type_to_prob_array[q_type]

    #if training_args.do_predict:
    assert training_args.training_type in ['FTonly', 'FTlmh', 'FTlpf', 'FTrubi', 'FT_trainedMask','FT_randMask']
    if model_args.zero_rate <= 0 or training_args.training_type == "FTonly" or training_args.training_type == "FTlmh" or training_args.training_type == "FTlpf" or training_args.training_type == "FTrubi":
#        mask_seed = model_args.mask_seed if 'rand' in model_args.struc_prun_type else ''
        if training_args.training_type == "FTonly" or training_args.training_type == "FTlmh"  or training_args.training_type == "FTlpf" or training_args.training_type == "FTrubi":
            print("We are fine-tunning the LXMERT without masker!")
        else:
            print("The mask rate is", model_args.zero_rate, "which is sammller than 0, equally training normal LXMERT like training_type 'FTonly'")
            assert training_args.training_type == "FTonly"

    else:
        if model_args.structured:
            print("The current study is under the settings of unstructured!")
            assert 0==1

            if model_args.prune_head:
                prune_with_mask(model, model_args.head_mask_dir, 'head')
            if model_args.prune_ffn:
                prune_with_mask(model, model_args.ffn_mask_dir, 'ffn')
        else:
            if training_args.training_type == "FT_trainedMask":

                assert model_args.prun_type == "mag"
                if model_args.mask_dir is not None:
                    print("model_args.mask_dir", model_args.mask_dir)
                    mask_dir = model_args.mask_dir
                    logger.info('\n\nLoading mask from %s'%mask_dir)
                    mask = torch.load(mask_dir)
                    if isinstance(mask,torch.nn.DataParallel):
                        mask = mask.module
                else:
                    print("Please give the mask_dir for loading the mask")
                    assert 0==1
                if model_args.clf_dir is not None:
                    clf_dir = model_args.clf_dir
                    clf_trainedMask = torch.load(clf_dir).cpu()
                    if isinstance(clf_trainedMask,torch.nn.DataParallel):
                        clf_trainedMask = clf_trainedMask.module
                    model.classifier = clf_trainedMask 
                else:
                    print("Please give the clf_dir (the classifier which is trainewed with mask traininf) for loading the classifier")
                    assert 0==1
                if model_args.model_type=='lxmert':
                    bert_model = model.lxmert

                pruning_model_with_mask(bert_model, mask, model_args.model_type) #这里对bert_model进行剪枝，本质上还是对model来剪，bert_model是model的地址，而不是一个copy
            
            elif training_args.training_type == "FT_randMask":
                assert model_args.prun_type == "rand"
                bert_model = model.lxmert
                mag_pruning(bert_model, model_args.zero_rate)
            else:
                print("If you wanna FT with mask, training_args.training_type should be in ['FT_trainedMask','FT_randMask']")
                assert 0==1

            zero = see_weight_rate(model, model_args.model_type)
            print('model 0:',zero)



    param_count = 0
    for n, p in model.named_parameters():
        param_count += p.nelement()
    param_count /= 1e6

    def compute_metrics(p: EvalPrediction) -> Dict:
        return compute_score_with_logits('vqa', p.predictions, p.label_ids)

    opt = init_optimizer(model, training_args, len(train_dataset)) if not model_args.structured else None
        # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        model_args=model_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=False,
        compute_metrics=compute_metrics,
        optimizers=opt
    )
    print("!!", trainer)

    fw_args = open(training_args.output_dir + '/args.txt', 'w')
    fw_args.write(str(training_args)+'\n\n')
    fw_args.write(str(model_args)+'\n\n')
    fw_args.write(str(data_args)+'\n\n')
    fw_args.write("Model size:%.2fM"%param_count+'\n\n')
    fw_args.close()


 

    # Training
    if training_args.do_train:
        _, _, best_score, results_at_best_score = trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        # trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        #del_model_command = 'rm -r %s/pytorch_model.bin'%training_args.output_dir
        #os.system(del_model_command)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)
        output_eval_file = os.path.join(
                training_args.output_dir, f"best_eval_results_vqa_noMASK.txt"
            )
        with open(output_eval_file, "w") as writer:
            logger.info("***** Best Eval results VQA noMASK*****")#.format(eval_dataset.args.task_name))
            for key, value in results_at_best_score.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))


    results = {}
    if training_args.do_predict:
        logger.info("*** Evaluate ***")

        eval_output = trainer.predict(eval_dataset)#evaluate()
        eval_dataloader = trainer.get_eval_dataloader(eval_dataset)
        json_str = make_json(eval_output[0],eval_output[3],eval_dataloader)
        with open(model_args.pred_out_dir+'/test.json' , 'w') as f:
            json.dump(json_str, f)
        result = {"eval_acc":eval_output[2]}
        output_eval_file = os.path.join(model_args.pred_out_dir, training_args.label4save+".txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    return results





def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
