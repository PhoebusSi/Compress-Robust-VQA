"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
from __future__ import print_function
from collections  import Counter
import os
import math
import json
from hg_transformers.tokenization_auto import AutoTokenizer
#from transformers import LxmertTokenizer, LxmertModel
import _pickle as cPickle
import numpy as np
import pickle
import utils_vqa as utils
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py
from xml.etree.ElementTree import parse
import torch
from torch.utils.data import Dataset
import zarr
import random
COUNTING_ONLY = False


# Following Trott et al. (ICLR 2018)
#   Interpretable Counting for Visual Question Answering
def is_howmany(q, a, label2ans):
    if 'how many' in q.lower() or \
            ('number of' in q.lower() and 'number of the' not in q.lower()) or \
                    'amount of' in q.lower() or \
                    'count of' in q.lower():
        if a is None or answer_filter(a, label2ans):
            return True
        else:
            return False
    else:
        return False


def answer_filter(answers, label2ans, max_num=10):
    for ans in answers['labels']:
        if label2ans[ans].isdigit() and max_num >= int(label2ans[ans]):
            return True
    return False


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                # the least frequent word (`bebe`) as UNK for Visual Genome dataset
                tokens.append(self.word2idx.get(w, self.padding_idx - 1))
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer):
    if None != answer:
        answer.pop('image_id')
        answer.pop('question_id')
    entry = {
        'question_id': question['question_id'],
        'image_id': question['image_id'],
        'image': img,
        'question': question['question'],
        'question_type': answer['question_type'],
        'answer': answer}

    return entry


def _load_dataset(dataroot, name, label2ans,ratio=1.0):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'test'
    """
    question_path = os.path.join(dataroot, 'vqacp_v2_%s_questions.json' % (name))
    questions = sorted(json.load(open(question_path)),
                           key=lambda x: x['question_id'])

    # train, val
    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))

    answers = sorted(answers, key=lambda x: x['question_id'])[0:len(questions)]

    utils.assert_eq(len(questions), len(answers))

    if ratio < 1.0:
        # sampling traing instance to construct smaller training set.
        index = random.sample(range(0,len(questions)), int(len(questions)*ratio))
        questions_new = [questions[i] for i in index]
        answers_new = [answers[i] for i in index]
    else:
        questions_new = questions
        answers_new = answers

    entries = []
    for question, answer in zip(questions_new, answers_new):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        if not COUNTING_ONLY or is_howmany(question['question'], answer, label2ans):
            entries.append(_create_entry(img_id, question, answer))


    return entries


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dataroot, image_dataroot, ratio, tokenizer, adaptive=False):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'test']
        self.name = name
        self.tokenizer = tokenizer

        ans2label_path = os.path.join(dataroot, 'cache', 'train_test_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'train_test_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        #self.dictionary = dictionary
        self.adaptive = adaptive

        print('loading image features and bounding boxes')

        with open("xxxx/CompressVQA/coco/object_features/vqa_img_feature_trainval.pickle", "rb") as f_f:
            f_f_data = pickle.load(f_f)

        self.features =  f_f_data

        if 1:

            self.entries = _load_dataset(dataroot, name, self.label2ans, ratio)

            self.tokenize()

            self.tensorize(name)

    def tokenize(self, max_length=14):
        tokenizer = self.tokenizer
        q_dict={}
        a_dict={}
        qa_dict={}
        K = 0

        for entry in self.entries:
            K = K + 1

            question_text = entry['question'] 
            question_type_text = entry['question_type'] 

            q_tokens = tokenizer._tokenize(question_text)

            length = len(q_tokens)
 



            if len(q_tokens) > max_length :
                q_tokens = q_tokens[:max_length]
                length = max_length
            else:
                padding = tokenizer._tokenize('[PAD]') * (max_length - len(q_tokens))
                q_tokens = q_tokens + padding

            q_tokens = [tokenizer._convert_token_to_id(token) for token in q_tokens]



            
            
            
            
            utils.assert_eq(len(q_tokens), max_length)
            entry['q_token'] = q_tokens
            entry['length'] = length
    def tensorize(self, name):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question
            length = torch.from_numpy(np.array(entry['length']))
            entry['length'] = length
              
            answer = entry['answer']
            if None != answer:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None





    def __getitem__(self, index):
        entry = self.entries[index]
        if not self.adaptive:
            features = torch.from_numpy(np.array(self.features[str(entry['image'])]['feats']))
            spatials = torch.from_numpy(np.array(self.features[str(entry['image'])]['sp_feats']))
            features = features.to(torch.float32)
            spatials = spatials.to(torch.float32)

        max_length = 14

        question = entry['q_token']
        length = entry['length']

        question_id = entry['question_id']
        image_id = entry['image_id']
        answer = entry['answer']
        #if self.name=="test":
        #    print(question_id)
        if None != answer:
            labels = answer['labels'] 
            scores = answer['scores'] 
            if None != scores:
                max_index = int(torch.argmax(scores))
                max_label = labels[max_index]
            else:
                max_label = torch.tensor(int(torch.randint(0,self.num_ans_candidates,(1,)) ))#如果为None，则随机一个答案为正确答案

            target = torch.zeros(self.num_ans_candidates)
            if labels is not None:
                target.scatter_(0, labels, scores)

            return  question, features, spatials, target, question_id,image_id, entry['bias'], max_label #, entry['rand_mask_id'], entry['length']
            
        else:
            
            return  question, features, spatials, question_id, image_id, entry['bias'], max_label

    def __len__(self):
        return len(self.entries)


if __name__ == '__main__':

    from torch.utils.data import DataLoader
    print('1')

    dataroot = '/root/VQA/data/vqacpv2/'
    img_root = '/root/VQA/data/coco/'
    print('2')
    tokenizer = AutoTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
    train_dset = VQAFeatureDataset('train',  dataroot, img_root, ratio=1.0, tokenizer=tokenizer, adaptive=False)
    print('3')

    loader = DataLoader(train_dset, 256, shuffle=True, num_workers=1, collate_fn=utils.trim_collate)

    for q, v, b, a, qid, vid in loader:

        print(a.shape)
