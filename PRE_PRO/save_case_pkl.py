# -*- coding: utf-8 -*-
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pickle
import sys
import numpy as np
import os
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


def geometric_series_sum(a, r, n):
    return a * (1.0 - pow(r, n)) / (1.0 - r)


def multi_scale(x, y, level):
    patch_size = 256

    offset = (patch_size / 2) * geometric_series_sum(1.0, 2.0, float(level))
    x = x - offset
    y = y - offset

    # 需確認是否倍率是2倍遞減
    x = int(x / pow(2, level))
    y = int(y / pow(2, level))

    x = 0 if x<0 else x
    y = 0 if y<0 else y

    return x, y


def read2patch(filename, x, y, level):
    level = int(level)
    slide = pyvips.Image.new_from_file(filename, page=level)

    slide_region = pyvips.Region.new(slide)

    x = int(x)
    y = int(y)
    x1 ,y1 = multi_scale(x,y,level)
    slide_fetch = slide_region.fetch(int(x1), int(y1), 256, 256)

    img = np.ndarray(buffer=slide_fetch,
                     dtype=np.uint8,
                     shape=[256, 256, slide.bands])

    return img


def get_data(pack):
    slide, slide_region, data = pack
    slide_data = slide_region.fetch(data[2], data[3], patch_size, patch_size)
    img = np.ndarray(buffer=slide_data,
                     dtype=np.uint8,
                     shape=[patch_size, patch_size, slide.bands])
    
    wsi_name = data[0].split('/')[-1].split('.')[0]
    gt_mask_path = os.path.join(config["mask_root_path"], f"{wsi_name}.tif")

    gt_mask = read2patch(gt_mask_path, data[2], data[3],level=0)
    tumor_label = 1 if np.any(gt_mask > 0) else 0
    data.append(tumor_label)

    if img.shape[2] == 4:
        img = img[:, :, 0:3]

    if detect_background(img.copy()):
        return 'non_bg', data
    else:
        return 'background', data


def extract_xy(datas):
    results = {
        'non_bg': [],  # 有label的區域
        'background': []  # 去背
    }
    # wsi_path, level, sx, sy
    slide = pyvips.Image.new_from_file(datas[0][0], page=int(datas[0][1]))
    #     slide = pyvips.Image.openslideload(datas[0][0], level=int(datas[0][2]))

    slide_region = pyvips.Region.new(slide)
    for data in datas:
        # [wsi_path, level, sx, sy]
        data = [data[0], data[1], int(data[2]), int(data[3])]
        label, result = get_data([slide, slide_region, data])
        results[label].append(result)

    return results


if __name__ == '__main__':
    case = sys.argv[1]
    print(case)
    wsi_root_path = config['wsi_root_path']
    data_pkl_path = config['data_pkl_path']
    level = 0
    patch_size = config['patch_size']
    stride_size = config['stride_size']

    pkl_file_path = config['data_pkl_path'] + f"/{case}/{case}.pkl"
    if os.path.exists(pkl_file_path):
        print(f"PKL 檔案已存在: {pkl_file_path}")
        print("跳過處理！")
        sys.exit(0)
    
    print(f"開始處理 case: {case}")
    print(f"將儲存至: {pkl_file_path}")

    data_list = []
    wsi_path = f'{wsi_root_path}/{case}.tif'

    slide = pyvips.Image.new_from_file(f'{wsi_root_path}/{case}.tif', page=level)

    start_pos = int(config['patch_size'] * pow(2, 2))

    for sy in range(start_pos, slide.height - start_pos, stride_size):
        for sx in range(start_pos, slide.width - start_pos, stride_size):
            data_list.append([wsi_path, level, sx, sy])

    total_data = None

    chunks = np.array_split(data_list, 360)
    with ProcessPoolExecutor() as e:
        for label_datas in tqdm(e.map(extract_xy, chunks), total=len(chunks)):
            if total_data is None:
                total_data = label_datas
            else:
                for k, v in label_datas.items():
                    total_data[k] += v

    print("Done!")
    patch_region = np.array(total_data['non_bg'])
    print(patch_region)
    print('patch_region: ', patch_region.shape)

    # 確保目錄存在
    os.makedirs(os.path.dirname(pkl_file_path), exist_ok=True)

    with open(pkl_file_path, 'wb') as w:
        pickle.dump(patch_region, w)

    print("Finish!")