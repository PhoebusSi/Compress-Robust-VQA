import json, random
from collections import OrderedDict, defaultdict, Counter

vqa_ocr_files = {'train': 'vqa_ocr_object/vqa_train_ocr.json', 'nominival': 'vqa_ocr_object/vqa_nominival_ocr.json', 'minival': 'vqa_ocr_object/vqa_minival_ocr.json'}
vqa_files = {'train': 'vqav2/v2_mscoco_train2014_annotations.json', 'val': 'vqav2/v2_mscoco_val2014_annotations.json'}
vqa_cp_files = {'train': 'vqacp/vqacp_v2_train_questions.json', 'test': 'vqacp/vqacp_v2_test_questions.json'}

# Load vqa-cp train, test ids
with open(vqa_cp_files['train'], 'r') as f:
    datas = json.load(f)
    train_ids = [d['question_id'] for d in datas]
with open(vqa_cp_files['test'], 'r') as f:
    datas = json.load(f)
    test_ids = [d['question_id'] for d in datas]
print(len(train_ids), len(test_ids))

# Load question type, answer type
qtypes, atypes = {}, {}
for split in vqa_files:
    with open(vqa_files[split]) as f:
        datas = json.load(f)['annotations']
        qtypes.update({d['question_id']: d['question_type'] for d in datas})
        atypes.update({d['question_id']: d['answer_type'] for d in datas})

# Load all vqa-ocr datas
vqa_ocr_datas = {}
for split in vqa_ocr_files:
    with open(vqa_ocr_files[split], 'r') as f:
        datas = json.load(f)
        datas = {d['question_id']: d for d in datas}
        vqa_ocr_datas.update(datas)
        print(split, len(datas))

# Split the data according to vqacp split
vqa_cp_ocr_datas = {}
for split, ids in zip(['train', 'test'], [train_ids, test_ids]):
    vqa_cp_ocr_datas[split] = [vqa_ocr_datas[id] for id in ids if id in vqa_ocr_datas]
vqa_cp_ocr_datas['val'] = random.sample(vqa_cp_ocr_datas['test'], 20000)    # randomly sample 20000 data as val set
print(len(vqa_cp_ocr_datas['train']), len(vqa_cp_ocr_datas['test']), len(vqa_cp_ocr_datas['val']))

# Compute bias
qtype_to_count = defaultdict(Counter)
for d in vqa_cp_ocr_datas['train']:
    qid = d['question_id']
    qtype = qtypes[qid]
    for answer in set(d['answer']):
        qtype_to_count[qtype][answer] += d['answer'].count(answer)
qtype_to_prob = {qtype: {a: qtype_to_count[qtype][a]/sum(qtype_to_count[qtype].values()) for a in qtype_to_count[qtype]} for qtype in qtype_to_count}

vqa_cp_ocr_datas['train_bias'] = []
for d in vqa_cp_ocr_datas['train']:
    qid = d['question_id']
    qtype = qtypes[qid]
    new_d = d.copy()
    new_d['bias'] = [qtype_to_prob[qtype][answer] for answer in d['answer']]
    vqa_cp_ocr_datas['train_bias'].append(new_d)

# Build label file
labels = {}
for split in ['val', 'test']:
    labels[split] = []
    for d in vqa_cp_ocr_datas[split]:
        qid = d['question_id']
        img_id = d['image'].replace('val2014_img/', '').replace('train2014/', '').replace('.jpg', '')
        label = {a: min(d['answer'].count(a)/3, 1) for a in d['answer']}
        label_data = {
            'answer_type': atypes[qid],
            'img_id': img_id,
            'label': label,
            'question_id': qid,
            'question_type': qtypes[qid],
            'sent': d['question']
        }
        labels[split].append(label_data)

# # Save
for split in vqa_cp_ocr_datas:
    with open(f'vqacp_ocr_object/{split}.json', 'w') as f:
        json.dump(vqa_cp_ocr_datas[split], f)
    if split in ['test', 'val']:
        with open(f'vqacp_ocr_object/{split}_labels.json', 'w') as f:
            json.dump(labels[split], f)