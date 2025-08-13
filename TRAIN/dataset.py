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

ImageFile.LOAD_TRUNCATED_IMAGES = True

# +
class TumorDataset(Dataset):
    def __init__(self, config,transform, type_str='train'):
        super().__init__()
        self.config = config
        self.transform = transform
        self.type_str = type_str

        self.data_list = config[type_str+'_list']
        
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(BASE_DIR)
        PROJECT_NAME = os.path.basename(PROJECT_ROOT)
        self.project = f"{PROJECT_NAME}_pseudo"
        
        self.datas = self.load_data_pkl(self.data_list)
        
        
        self.total_len = len(self.datas)
        print("patch num: ", self.total_len)


    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        img_high, img_low, label, case = self.get_data(idx)

        if self.transform is not None:
            state = torch.get_rng_state()
            img_high = self.transform(img_high) 
            torch.set_rng_state(state)
            img_low = self.transform(img_low)

        return img_high, img_low, label, case

    def load_data_pkl(self, case_list):
        print(self.project)
        data = []
        for case in case_list:
            if case == self.project:
                print("preparing dataset pseudo label")
                for pl_pkl in os.listdir(self.config["pseudo_label_path"]):
                    print(pl_pkl)
                    data_pkl = os.path.join(self.config["pseudo_label_path"], pl_pkl)
                    with open(data_pkl, 'rb') as f:
                        while True:
                            try:
                                data.append(pickle.load(f))
                            except EOFError:
                                break
            else:
                data_pkl = os.path.join(self.config['data_pkl_path'], case, f'{case}.pkl')
                with open(data_pkl, 'rb') as f:
                    while True:
                        try:
                            data.append(pickle.load(f))
                        except EOFError:
                            break
        if data:
            mix_data = np.concatenate(data)
            print(np.shape(mix_data))
            return mix_data
        else:
            print("Warning: No data loaded!")
            return np.array([])

    def geometric_series_sum(self, a, r, n):
        return a * (1.0 - pow(r, n)) / (1.0 - r)

    def multi_scale(self, x, y, level):
        patch_size = self.config['patch_size']

        offset = (patch_size / 2) * self.geometric_series_sum(1.0, 2.0, float(level))
        x = x - offset
        y = y - offset

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
        img_high = self.read2patch(self.datas[idx][0], self.datas[idx][2], self.datas[idx][3],level=0)
        img_low = self.read2patch(self.datas[idx][0], self.datas[idx][2], self.datas[idx][3],level=2)
        case = self.datas[idx][0].split('/')[-1].split('.')[0]
        if img_high.shape[2] == 4:
            img_high = img_high[:, :, 0:3]
            img_low = img_low[:, :, 0:3]

        return img_high, img_low, int(self.datas[idx][4]), case

