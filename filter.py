'''import os
import skimage.io as io
import torch
import torch.nn as nn
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from tf_adpt import GPT2LMHeadModel
from tqdm import tqdm
import random
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
from clip1 import clip
from clip1.clip import _transform
import math
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import skimage.io as io1
from PIL import Image
from PIL import ImageFile
import PIL.Image
from timm.models.layers import trunc_normal_
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from lr_scheduler import build_scheduler
from misc import *
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

sys.path.append('/home/v.silvio/diffusion-image-captioning-main/Paper2/DDCap-main/ImageCaptioning/captioning')
from ImageCaptioning.captioning.utils.rewards import get_scores, get_self_cider_scores
from collections import OrderedDict

from transformers import (
  DistilBertTokenizer, DistilBertForMaskedLM, DistilBertConfig,
  CLIPProcessor, CLIPModel as CLIP, CLIPConfig,
  activations, PreTrainedTokenizer
) 

torch.set_printoptions(profile="full")  

from captioneval.coco_caption.pycocotools.coco import COCO
from captioneval.coco_caption.pycocoevalcap.eval import COCOEvalCap
from captioneval.cider.pyciderevalcap.ciderD.ciderD import CiderD'

import json


dataset = json.load(open('/home/v.silvio/diffusion-image-captioning-main/Paper2/DDCap-main/MSCOCO_Caption/annotations/captions_val2014.json', 'r'))
#dataset = dataset.getAnns()

imgToAnns = {ann['image_id']: [] for ann in dataset['annotations']}
anns =      {ann['id']:       [] for ann in dataset['annotations']}



for ann in dataset['annotations']:
    imgToAnns[ann['image_id']] += [ann]
    anns[ann['id']] = ann


print(imgToAnns)
print(anns)

print(imgToAnns[203564][0]['caption'])

annotation = json.load(open("./MSCOCO_Caption/annotations/captions_val2014.json", "r"))["annotations"]'''


string = 'COCO_val2014_000000391895.jpg'
string = string.split("_")
print(string)

string = string[2]
print(string)

string = string.lstrip('0')
print(string)

string = string[0:-4]
print(string)