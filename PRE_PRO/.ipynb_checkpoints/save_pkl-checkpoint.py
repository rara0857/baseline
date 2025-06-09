# -*- coding: utf-8 -*-
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pickle
import sys
import numpy as np
import logging
# logging.basicConfig(level=logging.DEBUG)
import pyvips

sys.path.append(r'../')
from config import config


def detect_background(img):
    np.seterr(divide='ignore', invalid='ignore')
    sat = np.nan_to_num(
        1 - np.amin(img, axis=2) / np.amax(img, axis=2))
    pix_sat_count = (sat < 0.1).sum()
    all_pix_count = (sat > -1).sum()

    if pix_sat_count > (all_pix_count * 0.75):
        return False
    return True


def get_data(pack):
    slide, gt_mask, slide_region, mask_region, data = pack
    slide_data = slide_region.fetch(data[3], data[4], patch_size, patch_size)
    img = np.ndarray(buffer=slide_data,
                     dtype=np.uint8,
                     shape=[patch_size, patch_size, slide.bands])

    if img.shape[2] == 4:
        img = img[:, :, 0:3]

    # data = [wsi_path, mask_path, level, sx, sy]
    mask_data = mask_region.fetch(data[3], data[4], patch_size, patch_size)
    mask = np.ndarray(buffer=mask_data,
                      dtype=np.uint8,
                      shape=[patch_size, patch_size, gt_mask.bands])

    if detect_background(img.copy()):
        if 255 in mask:
            return 'tumor', data
        else:
            return 'normal', data
    else:
        return 'background', data


def extract_xy(datas):
    results = {
        'tumor': [],  # 有label的區域
        'normal': [],  # 沒有label的區域
        'background': []  # 去背
    }
    slide = pyvips.Image.new_from_file(datas[0][0], page=int(datas[0][2]))
#     slide = pyvips.Image.openslideload(datas[0][0], level=int(datas[0][2]))
    gt_mask = pyvips.Image.new_from_file(datas[0][1], page=int(datas[0][2]))
#     gt_mask = pyvips.Image.openslideload(datas[0][1], level=int(datas[0][2]))

    

    slide_region = pyvips.Region.new(slide)
    mask_region = pyvips.Region.new(gt_mask)
    for data in datas:
        data = [data[0], data[1], int(data[2]), int(data[3]), int(data[4])]
        # [wsi_path, mask_path, level, sx, sy]
        label, result = get_data([slide, gt_mask, slide_region, mask_region, data])
        results[label].append(result)

    return results

if __name__ == '__main__':
    wsi_root_path = config['wsi_root_path']
    mask_root_path = config['preprocess_save_path']
    data_pkl_path = config['data_pkl_path']
    level = config['level']
    patch_size = config['patch_size']
    stride_size = config['stride_size']
    stage_list = ['train', 'valid']
    stage_list = ['valid']
    print("************************")

    for stage in stage_list:
        data_list = []
        if stage == 'train':
            print("add train case...")
            case_list = config['train_list']
            print(case_list)
        elif stage == 'valid':
            print("add valid case...")
            case_list = config['val_list']
            print(case_list)

        for file in tqdm(case_list):
            datas = []
            wsi_path = f'{wsi_root_path}/{file}.tif'
            mask_path = f'{mask_root_path}/{file}/{file}_mask.tiff'

#             slide = pyvips.Image.openslideload(f'{wsi_root_path}/{file}.tif', level=0)
            slide = pyvips.Image.new_from_file(f'{wsi_root_path}/{file}.tif', page=level)

            start_pos = int(config['patch_size'] * pow(2, 2))

            for sy in range(start_pos, slide.height - start_pos, stride_size):
                for sx in range(start_pos, slide.width - start_pos, stride_size):
                    datas.append([wsi_path, mask_path, level, sx, sy])
            data_list.append(datas)

        total_data = None
        for d_list in data_list:
            chunks = np.array_split(d_list, 360)
            with ProcessPoolExecutor() as e:
                for label_datas in tqdm(e.map(extract_xy, chunks), total=len(chunks)):
                    if total_data is None:
                        total_data = label_datas
                    else:
                        for k, v in label_datas.items():
                            total_data[k] += v

        print("Done!")
        tumor_region = np.array(total_data['tumor'])
        normal_region = np.array(total_data['normal'])
        bg_region = np.array(total_data['background'])
        print('tumor_total: ', tumor_region.shape)
        print('normal_total: ', normal_region.shape)
        print('bg_total: ', bg_region.shape)

        with open(f'{data_pkl_path}/{stage}_KVGH.pkl', 'wb') as w:
            pickle.dump(tumor_region, w)
            pickle.dump(normal_region, w)
            pickle.dump(bg_region, w)

        print("Finish!")
