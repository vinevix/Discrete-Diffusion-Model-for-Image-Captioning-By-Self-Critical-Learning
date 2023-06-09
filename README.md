# Exploring Discrete Diffusion Models for Image Captioning.


## Official implementation for the paper ["Exploring Discrete Diffusion Models for Image Captioning"](https://arxiv.org/abs/2211.11694)


## Training prerequisites

[comment]: <> (Dependencies can be found at the [Inference notebook]&#40;https://colab.research.google.com/drive/1tuoAC5F4sC7qid56Z0ap-stR3rwdk0ZV?usp=sharing&#41; )
You can use [docker](https://hub.docker.com/r/zixinzhu/pytorch1.9.0). Also, you can create environment and install dependencies:
```
conda env create -f environment.yml
```
or
```
bash install_req.sh
```
or
```
pip install -r requirements.txt
```


## COCO training

Download [train_captions](https://drive.google.com/file/d/1D3EzUK1d1lNhD2hAvRiKPThidiVbP2K_/view?usp=sharing).

Download [training images](http://images.cocodataset.org/zips/train2014.zip) and [validation images](http://images.cocodataset.org/zips/val2014.zip) and unzip (We use Karpathy et el. split).

Download [oscar_split_ViT-B_32_train_512.pkl](https://drive.google.com/file/d/1CVsEQ5YRH3b6ZVRr7gY7ni7Ge1TgHvuM/view?usp=share_link)  in ./data/coco/
### Microsoft COCO
```
│MSCOCO_Caption/
├──annotations/
│  ├── captions_train2014.json
│  ├── captions_val2014.json
├──train2014/
│  ├── COCO_train2014_000000000009.jpg
│  ├── ......
├──val2014/ 
│  ├── COCO_val2014_000000000042.jpg
│  ├── ......
```

### Prepare evaluation
Change the work directory and set up the code of evaluation :
```
cd ./captioneval/coco_caption
bash ./get_stanford_models.sh
```
### Run

```
MKL_THREADING_LAYER=GPU  python -m torch.distributed.launch --nproc_per_node 8  train.py  --out_dir /results_diff --tag caption_diff_vitb16
```
If you want train the model with trainable clip, you can use the command:
```
MKL_THREADING_LAYER=GPU  python -m torch.distributed.launch --nproc_per_node 8  train_tclip.py  --out_dir /results_diff --tag caption_diff_vitb16
```
Please noting that we detach the gradients of [CLS] tokens during the training process of clip model. Because We observe that when the image encoder (clip) is trainable, the gradient backward of [CLS] tokens will damage the training of image encoder (clip).
## Citation
If you use this code for your research, please cite:
```
@article{zhu2022exploring,
  title={Exploring Discrete Diffusion Models for Image Captioning},
  author={Zhu, Zixin and Wei, Yixuan and Wang, Jianfeng and Gan, Zhe and Zhang, Zheng and Wang, Le and Hua, Gang and Wang, Lijuan and Liu, Zicheng and Hu, Han},
  journal={arXiv preprint arXiv:2211.11694},
  year={2022}
}
```


## Acknowledgments
This repository is heavily based on [CLIP](https://github.com/openai/CLIP), [CLIP_prefix_caption](https://github.com/rmokady/CLIP_prefix_caption) and [Hugging-faces](https://github.com/huggingface/transformers) repositories.
For training we used the data of [COCO dataset](https://cocodataset.org/#home). 


This code has been reused in order to apply a reinforcement learning technique called 'Self-critical learning': https://arxiv.org/abs/1612.00563





