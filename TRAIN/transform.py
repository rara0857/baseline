#!/usr/bin/env python
# coding: utf-8
# %%
import random
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image
import numpy as np
import os
from torchvision import transforms
from itertools import combinations_with_replacement


# %%
PARAMETER_MAX = 10

def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v, max_v, bias=0):                #調整顏色均衡 0.0產生黑白 1.0原始圖像
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v, max_v, bias=0):             #對比度
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Cutout(img, v, max_v, bias=0):
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return CutoutAbs(img, v)


def CutoutAbs(img, v, **kwarg):
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img)


def Identity(img, **kwarg):
    return img


def Invert(img, **kwarg):
    return PIL.ImageOps.invert(img)


def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)


def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v)


def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)

def ColorJitter(img, v, max_v, bias = 0):
    transform = transforms.Compose([
        transforms.ColorJitter(saturation=(0, 5), hue=(-0.1, 0.1))
        ])
    return transform(img)


# %%


augs = [(AutoContrast, None, None),
        (Brightness, 0.9, 0.05),
        (Color, 0.9, 0.05),
        (Contrast, 0.9, 0.05),
        (Equalize, None, None),
        (Identity, None, None),
        (Posterize, 4, 4),
        (Rotate, 30, 0),
        (Sharpness, 0.9, 0.05),
        (ShearX, 0.3, 0),
        (ShearY, 0.3, 0),
        (Solarize, 256, 0),
        (TranslateX, 0.3, 0),
        (TranslateY, 0.3, 0),
        (ColorJitter, None, None)]
       


# %%
def fixmatch_augment_pool():
    # FixMatch paper
    augs = [(AutoContrast, None, None),
            (Brightness, 0.9, 0.05),
            (Color, 0.9, 0.05),
            (Contrast, 0.9, 0.05),
            (Equalize, None, None),
            (Identity, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 0.9, 0.05),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (TranslateX, 0.3, 0),
            (TranslateY, 0.3, 0),
            (ColorJitter, None, None)]
    return augs


# %%
def remedy_setting(number_op):
    
    global comb
    global weight
    global number_operation
    
    number_operation = number_op
    comb = list(combinations_with_replacement("ABCDEFGHIJKLMNO",number_op))
    for x in range(len(comb)):
        comb[x] = list(comb[x])
        for y in range(len(comb[x])):
            comb[x][y] = ord(comb[x][y]) - 65
    weight = [1 for i in range(len(comb))]


# %%
def random_choose_op():
    
    global comb
    global weight

    ops = random.choices(comb, weights = weight)
    
    print('== random_choose_op : ' + str(ops[0]))
    # 使用絕對路徑儲存 aug.npy
    current_dir = os.path.dirname(os.path.abspath(__file__))
    aug_file = os.path.join(current_dir, 'aug.npy')
    np.save(aug_file, ops[0])
    return ops[0]


# %%
class transform_image_unlabel(object):
    def __init__(self):
        # 使用絕對路徑載入 aug.npy
        current_dir = os.path.dirname(os.path.abspath(__file__))
        aug_file = os.path.join(current_dir, 'aug.npy')
        index = np.load(aug_file)
        self.ops = []
        for i in index:
            self.ops.append(augs[i])
        random.shuffle(self.ops)
    def __call__(self, img):
        for op, max_v, bias in self.ops:
            v = np.random.randint(1, 10)
            img = op(img, v=v, max_v=max_v, bias=bias)
#         if random.random() < 0.5:
#             img = CutoutAbs(img, int(150*0.5))
        return img


# %%
class transform_image_train(object):
    def __init__(self):
        
        self.augment_pool = fixmatch_augment_pool()
    def __call__(self, img):
        global number_operation
        current_dir = os.path.dirname(os.path.abspath(__file__))
        aug_file = os.path.join(current_dir, 'aug.npy')
        if os.path.isfile(aug_file):
            if random.random() < 0.5:
                try:
                    index = np.load(aug_file)
                    random.shuffle(index)
                    self.ops = []
                    for i in index:
                        self.ops.append(augs[i])
                except:
                    print("bug bug bug numpy")
                    self.ops = random.choices(self.augment_pool, k = number_operation)
            else:
                self.ops = random.choices(self.augment_pool, k = number_operation)
        else:
            self.ops = random.choices(self.augment_pool, k = number_operation)
            
        for op, max_v, bias in self.ops:
            if random.random() < 0.5:
                v = np.random.randint(1, 10)
                img = op(img, v=v, max_v=max_v, bias=bias)
#         if random.random() < 0.5:
#             img = CutoutAbs(img, int(150*0.5))
        return img


# %%
def adjust_weight(ops, diff):
    
    global weight
    global comb
    index = comb.index(ops)
    diff_min = 0.1
    diff_max = 0.4
    diff_avg = (diff_min + diff_max) / 2
    w = (diff_avg / (abs(diff_avg - diff_avg) + diff_avg)) - (diff_avg / (abs(diff_max - diff_avg) + diff_avg))
    adjust = (diff_avg / (abs(diff - diff_avg) + diff_avg)) - (diff_avg / (abs(diff_max - diff_avg) + diff_avg))
    adjust *= (1/w)
    weight[index] += adjust
    weight[index] = max(weight[index], 0.01)



# %%
# remedy_setting(3)
# ops = random_choose_op()

# %%
# adjust_weight(ops, 0.8)

# %%
def check_weight():
    global weight
    print(weight)
