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
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, CTRL, BERT, RoBERTa, XLNet).
GPT, GPT-2 and CTRL are fine-tuned using a causal language modeling (CLM) loss. BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss. XLNet is fine-tuned using a permutation language modeling (PLM) loss.
"""


import logging
from torch.autograd import Variable
from collections import defaultdict, Counter
import math
import os 
from dataclasses import dataclass, field
from typing import Optional
import masking.sparsity_control_Robust as sp_control
import masking.maskers_visualBert as maskers

import utils.param_parser as param_parser
import torch.nn.utils.prune as prune
import torch
import numpy as np
import torch.nn as nn
from optimization import AdamW
import utils4VQA
from dataset_LXM import Dictionary, VQAFeatureDataset

from hg_transformers.optimization import get_linear_schedule_with_warmup, get_constant_schedule, get_constant_schedule_with_warmup

from hg_transformers.mask_trainer_visualBERT_VQA import Trainer


from hg_transformers import TrainingArguments as BaseTrainingArguments
from hg_transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    TrimCollator,
    HfArgumentParser,
    PreTrainedTokenizer,
    set_seed,
)
from hg_transformers.trainer_utils import EvalPrediction
from typing import Dict, Optional
from hg_transformers.data.metrics import compute_score_with_logits



logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


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
    para_num_dict = {}
    for ii in range(12):
        sum_list = sum_list+float(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.query.weight_mask'].nelement())
        para_num_dict['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.query.weight_mask'] = float(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.query.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.query.weight_mask'] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.key.weight_mask'].nelement())
        para_num_dict['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.key.weight_mask'] = float(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.key.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.key.weight_mask'] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.value.weight_mask'].nelement())
        para_num_dict['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.value.weight_mask'] = float(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.value.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.self.value.weight_mask'] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.output.dense.weight_mask'].nelement())
        para_num_dict['%s.encoder.layer.'%model_type+str(ii)+'.attention.output.dense.weight_mask'] = float(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.output.dense.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.attention.output.dense.weight_mask'] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.intermediate.dense.weight_mask'].nelement())
        para_num_dict['%s.encoder.layer.'%model_type+str(ii)+'.intermediate.dense.weight_mask']=float(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.intermediate.dense.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.intermediate.dense.weight_mask'] == 0))

        sum_list = sum_list+float(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.output.dense.weight_mask'].nelement())
        para_num_dict['%s.encoder.layer.'%model_type+str(ii)+'.output.dense.weight_mask']=float(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.output.dense.weight_mask'].nelement())
        zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.encoder.layer.'%model_type+str(ii)+'.output.dense.weight_mask'] == 0))

    

    sum_list = sum_list+float(model.state_dict()['%s.pooler.dense.weight_mask'%model_type].nelement())
    para_num_dict['%s.pooler.dense.weight_mask'%model_type]=float(model.state_dict()['%s.pooler.dense.weight_mask'%model_type].nelement())
    zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.pooler.dense.weight_mask'%model_type] == 0))
    sum_list = sum_list+float(model.state_dict()['%s.embeddings.word_embeddings.weight_mask'%model_type].nelement())
    para_num_dict['%s.embeddings.word_embeddings.weight_mask'%model_type]=float(model.state_dict()['%s.embeddings.word_embeddings.weight_mask'%model_type].nelement())
    zero_sum = zero_sum+float(torch.sum(model.state_dict()['%s.embeddings.word_embeddings.weight_mask'%model_type] == 0))

    print("model.state_dict()", model.state_dict().keys())
    return 100*zero_sum/sum_list, para_num_dict, sum_list, zero_sum




def init_masker(conf, model, logger):
    # init the masker scheduler.

    conf.masking_scheduler_conf_ = (
        param_parser.dict_parser(conf.masking_scheduler_conf)
        if conf.masking_scheduler_conf is not None
        else None
    )
    conf.masking_scheduler_conf_['final_sparsity'] = conf.zero_rate
    if conf.masking_scheduler_conf is not None:
        for k, v in conf.masking_scheduler_conf_.items():
            setattr(conf, f"masking_scheduler_{k}", v)
    conf.logger = logger

    masker_scheduler = sp_control.MaskerScheduler(conf)



    weight_types = ["K", "Q", "V", "AO", "I", "O", "P", "E"] 



    
    # init the masker.
    masker = maskers.Masker(
        masker_scheduler=masker_scheduler,
        logger=logger,
        mask_biases=conf.mask_biases,
        structured_masking_info={
            "structured_masking": conf.structured_masking,
            "structured_masking_types": conf.structured_masking_types,
            "force_masking": conf.force_masking,
        },
        threshold=conf.threshold,
        init_scale=conf.init_scale,
        which_ptl=conf.model_type,
        controlled_init=conf.controlled_init,
        # global_prune=conf.global_prune,
    )
    print('masker', masker)

    # parse the get the names of layers to be masked.
    assert conf.layers_to_mask is not None, "Please specify which BERT layers to mask."
    conf.layers_to_mask_ = (
        [int(x) for x in conf.layers_to_mask.split(",")]
        if "," in conf.layers_to_mask
        else [int(conf.layers_to_mask)]
    )
    names_tobe_masked = set()
    # names_tobe_masked, name_in_modal, name_in_module, name_in_layer = maskers.chain_module_names(
    names_tobe_masked = maskers.chain_module_names(
        conf.model_type, conf.layers_to_mask_, weight_types
    )
    # print("name_in_modal", name_in_modal)
    if conf.mask_classifier:
        if conf.type not in ["lxmert", "visual_bert"]:
            print("Cont.type should be set to lxmert or visual_bert !")
            assert 1==2
        names_tobe_masked.add("classifier")
        name_in_module["classifier"] = "classifier"

    masker.patch_modules(
        model=model,
        names_tobe_masked=names_tobe_masked,
        name_of_masker=conf.name_of_masker,
    )
    return masker


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
        default="Masker", metadata={"help": "Masker"}
    )
    gamma: Optional[float] = field(
        default=5, metadata={"help": "gamma for LPF loss"}
    )

    #the training methods (w/ or w/o debiasing methods) during mask train
    Masker_type:Optional[str] = field(
        default=None, metadata={"help": "normal, lmh, lpf, rubi"}
    )
    #This option determines the path of the loaded model
    FTmodel_type:Optional[str] = field(
        default=None, metadata={"help": "noFT, normal, lmh, lpf, rubi"}
    )
    label4save: Optional[str] = field(
        default=None, metadata={"help": "FT model method + masker methoddebiasing + continueFT/or not, e.g., normal+lmhMasker+continueFT, normal+lmh_Masker, normal, lmh, lpf, rubi"}
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
    masker_level: Optional[str] = field(
        default="modal", metadata={"help": "modal ; module ; layer"}
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default="/mnt/bn/qingyi-bn-lq/data4OKVQA/model/bert-base-uncased", metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    zero_rate: Optional[float] = field(
        default=0., metadata={"help": "The percentate of 0 in model weights."}
    )
    # Lang_comp: Optional[float] = field(
    #     default=0., metadata={"help": "The percentate of 0 in model's language-modal modules weights."}
    # )
    # Vis_comp: Optional[float] = field(
    #     default=0., metadata={"help": "The percentate of 0 in model's visual-modal modules weights."}
    # )
    # Fus_comp: Optional[float] = field(
    #     default=0., metadata={"help": "The percentate of 0 in model's fusion-modal modules weights."}
    # )
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
    global_prune: str2bool = field(
        default=False, metadata={"help": "Whether to prune global!"}
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
    pred_model_dir: Optional[str] = field(
        default=None, metadata={"help": "The model which is used for prediction and output json files."}
    )
    pred_out_dir: Optional[str] = field(
        default=None, metadata={"help": "The dir where the json files will save."}
    )
    #the path of pretrained models obtaned in Stage1 (training with BCE: FTvisbert_dir; training with LMH: lmhFTvisbert_dir; training with LPF: lpfFTvisbert_dir; training with rubi: rubiFTvisbert_dir)
    FTlxmert_dir: Optional[str] = field(
    FTvisbert_dir: Optional[str] = field(
        default="/mnt/bn/qingyi-bn-lq/CompressVQA/pretrained_model/visualBERT_pre", metadata={"help": "The model fine-tuned normmally directory."}
    )
    lmhFTvisbert_dir: Optional[str] = field(
        default="/mnt/bn/qingyi-bn-lq/CompressVQA/pretrained_model/visualBERTLMH", metadata={"help": "The model fine-tuned with LMH  directory."}
    )
    lpfFTvisbert_dir: Optional[str] = field(
        default="/mnt/bn/qingyi-bn-lq/CompressVQA/pretrained_model/visualBERTLPF", metadata={"help": "The model fine-tuned with LPF  directory."}
    )
    rubiFTvisbert_dir: Optional[str] = field(
        default="/mnt/bn/qingyi-bn-lq/CompressVQA/pretrained_model/visualBERTRUBI", metadata={"help": "The model fine-tuned with rubi  directory."}
    ) 

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """


    dataroot: Optional[str] = field(
        default='xxx/CompressVQA/vqacpv2/', metadata={"help": "The input text training data file (a text file)."}

    )
    img_root: Optional[str] = field(
        default='xxx/CompressVQA/coco/', metadata={"help": "The input img training data file (a text file)."}

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
        default=1.0, metadata={"help": "Ratio of training set used"}
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


def init_mask_weight(args, device, component_type):
    """
    component_type = 'head' or 'ffn'
    """
    mask_dir = args.load_head_mask_dir if component_type=='head' else args.load_ffn_mask_dir
    logger.info("Loading %s score from %s"%(component_type, mask_dir))
    score = torch.from_numpy(np.load(mask_dir))
    # Rescale
    #score =(score - score.min()) / (score.max() - score.min()) * args.init_scale
    _bool_masks = (
        score
        > torch.kthvalue(input=score.view(-1), k=int(score.numel()*args.zero_rate)).values
    )
    score[_bool_masks] = 2.0 * 1e-2
    score[~_bool_masks] = 0.0 * 1e-2
    mask_weight = torch.nn.Parameter(score.to(device))
    return mask_weight


def get_dataset(args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, evaluate=False):
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
    optimizer = AdamW(params, lr= training_args.learning_rate, eps= training_args.adam_epsilon)
    num_training_steps = int(int(num_train_data/(training_args.n_gpu*\
                     training_args.per_gpu_train_batch_size)+1)*training_args.num_train_epochs)
    scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=num_training_steps)
    opt = (optimizer,scheduler)
    return opt

class Binarizer_head(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, num_to_mask, num_heads):
        return binarizer_fn_head(inputs, num_to_mask, num_heads)

    @staticmethod
    def backward(ctx, gradOutput):
        return (gradOutput, None, None)

def binarizer_fn_head(inputs, num_to_mask, num_heads):
    outputs = inputs.clone()
    heads_to_mask = outputs.view(-1).sort()[1]
    outputs[:,:] = 1.0
    for i, head in enumerate(heads_to_mask):
        if i==num_to_mask:
            break
        layer_idx = head.item() // num_heads
        head_idx = head.item() % num_heads
        outputs[layer_idx][head_idx] = 0.0
    return outputs

class Binarizer_ffn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, threshold):
        return binarizer_fn_ffn(inputs, threshold)

    @staticmethod
    def backward(ctx, gradOutput):
        return (gradOutput, None, None)

def binarizer_fn_ffn(inputs, threshold):
    outputs = inputs.clone()
    outputs[inputs.le(threshold)] = 0.0
    outputs[inputs.gt(threshold)] = 1.0
    return outputs

def freeze_params(model, args):
    for name, param in model.named_parameters():
        if (not 'predictions' in name) or args.freeze_mlm_head:
            param.requires_grad = False
    return model

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

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    """
    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )
    """
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

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        print(model_args.model_name_or_path)
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    if training_args.use_kd:
        config.output_hidden_states = True

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir=model_args.cache_dir)

    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if training_args.do_predict:
        print("This script ar not support for predicting only!")
        assert 1==2
    if not training_args.do_train:
        print("do_train in this process has to be True!!")
        assert 1==2
    if training_args.do_train:
        if training_args.FTmodel_type == 'noFT':
            if not model_args.model_name_or_path:
                print("pleaze set the model_args.model_name_or_path for loading the noFT model to training Masker")
                assert 0==1
            else:
                model = AutoModelForMultipleChoice.from_pretrained(
                    model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=config,
                    cache_dir=model_args.cache_dir,
                )
        elif training_args.FTmodel_type == 'normal':
            if not model_args.FTvisbert_dir:
                print("pleaze set the model_args.FTvisbert_dir for loading the FT model(normmal) to training Masker")
                assert 0==1
            else:
                print(model_args.FTvisbert_dir+'/FTonlyNoCompress_FTonly.bin')
                model = torch.load(model_args.FTvisbert_dir+'/FTonlyNoCompress_FTonly.bin').cpu()
                if isinstance(model,torch.nn.DataParallel):
                    model = model.module


        elif training_args.FTmodel_type == 'lmh':
            if not model_args.lmhFTvisbert_dir:
                print("pleaze set the model_args.model_name_or_path for loading the model finetuned with LMH to training Masker")
                assert 0==1
            else:
                model = torch.load(model_args.lmhFTvisbert_dir+'/FTlmh_NoCompress_FTlmh_only.bin').cpu()
                if isinstance(model,torch.nn.DataParallel):
                    model = model.module
        elif training_args.FTmodel_type == 'lpf':
            if not model_args.lpfFTvisbert_dir:
                print("pleaze set the model_args.model_name_or_path for loading the model finetuned with LPF to training Masker")
                assert 0==1
            else:
                model = torch.load(model_args.lpfFTvisbert_dir+'/FTlpf_NoCompress_FTlpf_only.bin').cpu()
                if isinstance(model,torch.nn.DataParallel):
                    model = model.module
        elif training_args.FTmodel_type == 'rubi':
            if not model_args.rubiFTvisbert_dir:
                print("pleaze set the model_args.model_name_or_path for loading the model finetuned with RUBI to training Masker")
                assert 0==1
            else:
                model = torch.load(model_args.rubiFTvisbert_dir+'/FTrubi_NoCompress_FTrubi_only.bin').cpu()
                #model = model.to(training_args.device)
                if isinstance(model,torch.nn.DataParallel):
                    model = model.module    
        else:
            print("training_args.FTmodel_type should be in noFT, normal, lmh, lpf, rubi ")
            assert 0==1

    
    model.resize_token_embeddings(len(tokenizer))
    
    

    if training_args.use_kd:
        teacher_tokenizer = AutoTokenizer.from_pretrained(model_args.teacher_model, cache_dir=model_args.cache_dir)
        teacher_config = AutoConfig.from_pretrained(model_args.teacher_model, cache_dir=model_args.cache_dir)
        teacher_config.output_hidden_states = True
        teacher_model = AutoModelWithLMHead.from_pretrained(
            model_args.teacher_model,
            from_tf=bool(".ckpt" in model_args.teacher_model),
            config=teacher_config,
            cache_dir=model_args.cache_dir,
        )
        teacher_model.resize_token_embeddings(len(teacher_tokenizer))
    else:
        teacher_model = None


    if model_args.structured:
        print("unstructrued only in the work!!!")
        assert 0==1
        head_mask_weight, ffn_mask_weight = None, None
        if model_args.train_head_mask:
            head_mask_weight = init_mask_weight(model_args, training_args.device, 'head')
        if model_args.train_ffn_mask:
            ffn_mask_weight = init_mask_weight(model_args, training_args.device, 'ffn')
        masker = None
        model = freeze_params(model, model_args)
    else:

        print('model', model)

        masker = init_masker(model_args, model, logger) #, hpmodel, model_args)


        zero, para_num_dict, para_num, zero_num = see_weight_rate(model, model_args.model_type)#统计剪枝的比例的，需要改成lxmert对应的！
        print('此处看到的是没有压缩前的model的各个模块的参数量和总参数量:',zero, para_num, zero_num)
        for mo in para_num_dict.keys():
            print(mo+" 参数数量为：", para_num_dict[mo] )
        print("0的数量就是被压缩掉的数量\n\n\n\n")


        head_mask_weight = None
        ffn_mask_weight = None

    if config.model_type in ["bert", "roberta", "distilbert", "camembert"] and not data_args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the"
            "--mlm flag (masked language modeling)."
        )


    # Get datasets
    train_dataset = get_dataset(data_args, tokenizer=tokenizer) #if training_args.do_train else None
    eval_dataset = get_dataset(data_args, tokenizer=tokenizer, evaluate=True) #if training_args.do_eval else None
 
    #在加上LMH使用这里/
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
    
    
    
    

    data_collator = TrimCollator()
    
    opt = init_optimizer(model, training_args, len(train_dataset)) if not model_args.structured else None

    threshold_fn_head = Binarizer_head().apply if model_args.structured else None
    threshold_fn_ffn = Binarizer_ffn().apply if model_args.structured else None

    
    
    def compute_metrics(p: EvalPrediction) -> Dict:
        return compute_score_with_logits('vqa', p.predictions, p.label_ids)
    


    # Initialize our Trainer
    trainer = Trainer(
        model=model, 
        teacher_model=teacher_model,
        args=training_args,
        model_args=model_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=False,
        compute_metrics=compute_metrics,
        optimizers=opt,
        masker=masker,
        head_mask_weight=head_mask_weight,
        ffn_mask_weight=ffn_mask_weight,
        threshold_fn_head=threshold_fn_head,
        threshold_fn_ffn=threshold_fn_ffn
    )



    fw_args = open(training_args.output_dir + '/args.txt', 'w')
    fw_args.write(str(training_args)+'\n\n')
    fw_args.write(str(model_args)+'\n\n')
    fw_args.write(str(data_args)+'\n\n')
    fw_args.close()


    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )

        _, _, _, results_at_best_score = trainer.train( model_path=model_path) 

    
    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output, _ = trainer.evaluate()


        result = {"eval_acc":eval_output["eval_acc"]}
        output_eval_file = os.path.join(training_args.output_dir, "eval_results_vqa.txt")
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
