# -*- coding: utf-8 -*-
import os
from os.path import join
import torch
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from concurrent.futures import ProcessPoolExecutor
from PIL import ImageFile
import pickle
import pyvips
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


class   TumorDataset(Dataset):
    def __init__(self, case, config, transform, pkl=False):
        super().__init__()
        self.config = config
        self.transform = transform
        self.read_wsi(case)

        data_pkl_path = self.config['data_pkl_path'] + f"{case}/{case}.pkl"
        save_dir = os.path.dirname(data_pkl_path)

        self.patch_list = self.load_data_pkl(data_pkl_path)
        self.total_len = len(self.patch_list)
        print("patch num: ", self.total_len)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        img_high, img_low, x, y, label = self.get_data(idx)

        if self.transform is not None:
            state = torch.get_rng_state()
            img_high_1 = self.transform(img_high)
            torch.set_rng_state(state)
            img_low_1 = self.transform(img_low)
            
            state = torch.get_rng_state()
            img_high_2 = self.transform(img_high)
            torch.set_rng_state(state)
            img_low_2 = self.transform(img_low)
            
            state = torch.get_rng_state()
            img_high_3 = self.transform(img_high)
            torch.set_rng_state(state)
            img_low_3 = self.transform(img_low)

        return img_high_1, img_low_1, img_high_2, img_low_2, img_high_3, img_low_3, x, y, label



    def read2patch(self, filename, x, y, level):
        level = int(level)

        if level == 0:
            slide_region = pyvips.Region.new(self.slide)
        else: # low level
            slide_region = pyvips.Region.new(self.slide_low)

        x = int(x)
        y = int(y)
        x1, y1 = self.multi_scale(x, y, level)
        slide_fetch = slide_region.fetch(int(x1), int(y1), self.config['patch_size'], self.config['patch_size'])

        img = np.ndarray(buffer=slide_fetch,
                         dtype=np.uint8,
                         shape=[self.config['patch_size'], self.config['patch_size'], 3])

        return img

    def get_data(self, idx):
        # data = [wsi_path, level, sx, sy] for inference, [wsi_path, level, sx, sy, label] for training
        # 確保座標是整數類型
        x_coord = int(self.patch_list[idx][2])
        y_coord = int(self.patch_list[idx][3])
        
        img_high = self.read2patch(self.patch_list[idx][0], x_coord, y_coord, level=0)
        img_low = self.read2patch(self.patch_list[idx][0], x_coord, y_coord, level=2)

        if img_high.shape[2] == 4:
            img_high = img_high[:, :, 0:3]
            img_low = img_low[:, :, 0:3]

        # Now pkl files should contain labels (either ground truth or -1 for unknown)
        return img_high, img_low, x_coord, y_coord, int(self.patch_list[idx][4])

    def read_wsi(self,case):
        wsi_path = self.config['wsi_root_path'] + f'{case}.tif'
        mask_path = self.config["mask_path"]+f"{case}.tif"
        self.slide =pyvips.Image.new_from_file(wsi_path, page = 0) # high level
        self.slide_low = pyvips.Image.new_from_file(wsi_path, page = 2) # low level
        self.mask = pyvips.Image.new_from_file(mask_path, page = 0)

    def load_data_pkl(self, data_pkl):
        print('Load data pkl from: ', data_pkl)
        data = []
        with open(data_pkl, 'rb') as f:
            while True:
                try:
                    data.append(pickle.load(f))
                except EOFError:
                    break
        return data[0]

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

        return x, y
