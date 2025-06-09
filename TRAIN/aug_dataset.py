# -*- coding: utf-8 -*-
import os
from os.path import join
import torch
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from PIL import ImageFile
import pickle
import pyvips
from transform import transform_image_unlabel
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


# +
class TumorDataset(Dataset):
    def __init__(self, config,transform, type_str='train'):
        super().__init__()
        # print(self.datas)
        self.config = config
        self.identity_transform = transform
        self.type_str = type_str

        self.data_list = config[type_str+'_list']
        self.datas = self.load_data_pkl(self.data_list)
        
        
        self.total_len = len(self.datas)
        print("patch num: ", self.total_len)


    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        img_high, img_low, label, case = self.get_data(idx)
        
        self.aug_transform = transforms.Compose([
            transforms.ToPILImage(),
            transform_image_unlabel(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])


        state = torch.get_rng_state()
        img_high_aug = self.aug_transform(img_high) 
        torch.set_rng_state(state)
        img_low_aug = self.aug_transform(img_low)
        
        img_high_identity = self.identity_transform(img_high)
        img_low_identity = self.identity_transform(img_low)

        return img_high_identity, img_low_identity, img_high_aug, img_low_aug, label, case

    def load_data_pkl(self, case_list):
        data = []
        for case in case_list:
            data_pkl = self.config['data_pkl_path'] + f'/{case}/{case}.pkl'
            with open(data_pkl, 'rb') as f:
                while True:
                    try:
                        data.append(pickle.load(f))
                    except EOFError:
                        break

        # data 0: tumor, 1: normal, 2: background
        mix_data = np.concatenate(data)
        print(np.shape(mix_data))
        return mix_data

    def geometric_series_sum(self, a, r, n):
        return a * (1.0 - pow(r, n)) / (1.0 - r)

    def multi_scale(self, x, y, level):
        patch_size = self.config['patch_size']

        offset = (patch_size / 2) * self.geometric_series_sum(1.0, 2.0, float(level))
        x = x - offset
        y = y - offset

        # 需確認是否倍率是2倍遞減
        x = int(x / pow(2, level))
        y = int(y / pow(2, level))
        
        x = 0 if x<0 else x
        y = 0 if y<0 else y
        
        return x, y

    def read2patch(self, filename, x, y, level):
        level = int(level)
        slide = pyvips.Image.new_from_file(filename, page=level)

        slide_region = pyvips.Region.new(slide)
        
        x = int(x)
        y = int(y)
        x1 ,y1 = self.multi_scale(x,y,level)
        slide_fetch = slide_region.fetch(int(x1), int(y1), self.config['patch_size'], self.config['patch_size'])

        img = np.ndarray(buffer=slide_fetch,
                         dtype=np.uint8,
                         shape=[self.config['patch_size'], self.config['patch_size'], slide.bands])

        return img

    def get_data(self, idx):
        # data = [wsi_path, level, sx, sy, label]
        img_high = self.read2patch(self.datas[idx][0], self.datas[idx][2], self.datas[idx][3],level=0)
        img_low = self.read2patch(self.datas[idx][0], self.datas[idx][2], self.datas[idx][3],level=2)
        case = self.datas[idx][0].split('/')[-1].split('.')[0]
        if img_high.shape[2] == 4:
            img_high = img_high[:, :, 0:3]
            img_low = img_low[:, :, 0:3]

#         gt_mask_case = os.path.basename(self.datas[idx][0]).split('.')[0]
#         gt_mask_path = self.config['data_pkl_path']+f'/{gt_mask_case}/{gt_mask_case}_mask.tiff'
#         gt_mask = self.read2patch(gt_mask_path, self.datas[idx][2], self.datas[idx][3],level=0)
#         if 255 in gt_mask:
#             label = 1
#         else:
#             label = 0
        
        return img_high, img_low, int(self.datas[idx][4]), case

# -





