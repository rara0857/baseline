import os, sys
import torch
from config_test import config
from model import net_20x_5x
from evaluation import *
import torch.nn as nn
import time
import csv
from dataloader_pyvips import TumorDataModule

def read_file_list(filename):
    file_list = []
    with open(filename, 'r') as f:
        for line in f:
            file_list.append(line.strip())
    return file_list

case_list = read_file_list('../test_list.txt')

# 自動檢測可用的 GPU 數量
available_gpus = torch.cuda.device_count()
print(f"Available GPUs: {available_gpus}")

if available_gpus > 1:
    cuda_devices = "0,1"
elif available_gpus == 1:
    cuda_devices = "0"
else:
    print("No CUDA devices available, using CPU")
    cuda_devices = ""

if cuda_devices:
    device_ids = [int(d) for d in cuda_devices.split(',')]
    main_device = device_ids[0]
else:
    device_ids = []
    main_device = 0

def run_inference(model, case, device):
    from evaluation import init_para, save_evaluation, show_testing_img
    dataloader = TumorDataModule(config, case=case).test_dataloader()
    model.eval()
    softmax_layer = nn.Softmax(dim=1)
    init_para(case)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            img_high_1, img_low_1, img_high_2, img_low_2, img_high_3, img_low_3, x, y, label = batch
            img_high_1, img_low_1 = img_high_1.to(device), img_low_1.to(device)
            
            # 使用 autocast 進行混合精度推理
            if torch.cuda.is_available():
                with torch.autocast(device_type='cuda'):
                    prob = model(img_high_1, img_low_1)
            else:
                prob = model(img_high_1, img_low_1)
            softmax_output = softmax_layer(prob)
            show_testing_img(prob, x, y)
            
    IOU_score, F1_score = save_evaluation(case)
    return IOU_score, F1_score

num_class = config['num_class']
model = net_20x_5x(num_classes=num_class)

checkpoint_path = os.path.join(config['log_string_dir'], config['best_weights'])
print(f"Loading checkpoint from: {checkpoint_path}")

# 設定裝置
if torch.cuda.is_available() and device_ids:
    device = torch.device(f"cuda:{main_device}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = nn.DataParallel(model, device_ids=device_ids)
    print(f"Using GPU devices: {device_ids}")
else:
    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    print("Using CPU for inference")

try:
    model.load_state_dict(checkpoint['model_state_dict'])
except RuntimeError:
    # 處理 DataParallel 包裝的模型
    if torch.cuda.is_available() and device_ids:
        new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model_state_dict'].items()}
        model.module.load_state_dict(new_state_dict)
    else:
        # CPU 情況下，移除 module. 前綴
        new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model_state_dict'].items()}
        model.load_state_dict(new_state_dict)

model.to(device)
model.eval()

with open('result.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'iou', 'f1', 'costtime'])
    for case in case_list:
        t_start = time.time()
        IOU_score, F1_score = run_inference(model, case, device)
        cost_time = time.time() - t_start
        writer.writerow([case, IOU_score, F1_score, cost_time])
        print(f"{case} IOU: {IOU_score:.4f}  F1: {F1_score:.4f}  Cost time: {cost_time:.2f}s")