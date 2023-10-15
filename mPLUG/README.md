# mPLUG with mask training

## Prepare
### Download Models
- Download [bert-base-uncased](https://huggingface.co/bert-base-uncased) to `ckpts/bert-base-uncased/`
- Download [ViT-B-16.tar](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/mPLUG/ViT-B-16.tar) to `ckpts/`
- Download [mplug_base.pth](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/mPLUG/mplug_base.pth) to `ckpts/`

### Download and process data
- Download [VQA-CP v2 data](https://www.iro.umontreal.ca/~agrawal/vqa-cp/) to `data/vqacp/`
- Download the [processed VQA v2 data](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/mPLUG/data.tar) to `data/vqacp_ocr_object/`
- Download the coco [train](http://images.cocodataset.org/zips/train2014.zip) and [val](http://images.cocodataset.org/zips/train2014.zip) images to `img_root/coco_2014/`
- Process the VQA-CP v2 data by running `python build_vqacp_ocr.py`

## Requirements
* [PyTorch](https://pytorch.org/) version >= 1.11.0

* Install other libraries via
```
pip install -r requirements.txt
```

## Training
- mPLUG(CE)
  `bash scripts/vqa_full_model.sh`
- mPLUG(LPF)
  `bash scripts/vqa_full_model_debias.sh`
- mPLUG(CE)+mask train(LPF)
  `bash scripts/vqa_full+mask_debias.sh`
- mPLUG(LPF)+mask train(LPF)
  `bash scripts/vqa_full_debias_mask_debias.sh`

## Results
| Model       |  Acc  |  Param(%) |
|-------------|:-----:|:-----:|
| mPLUG(CE)                       | 57.05  | 100% |
| mPLUG(CE)+mask train(LPF)       | 62.53  | 51.98% |
| mPLUG(LPF)                      | 65.24  | 100% |
| mPLUG(LPF)+mask train(LPF)      | 63.66  | 51.98% |
