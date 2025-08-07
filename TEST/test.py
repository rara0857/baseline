import os, sys
import torch
from config_test import config
from model import net_20x_5x
from evaluation import *
import torch.nn as nn
from torch.cuda.amp import autocast
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

cuda_devices = "0,1"  # 可以設定多個GPU，例如 "0,1,2,3" 或 "0"
device_ids = [int(d) for d in cuda_devices.split(',')]
main_device = device_ids[0]

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
            
            with autocast():
                prob = model(img_high_1, img_low_1)
            softmax_output = softmax_layer(prob)
            show_testing_img(prob, x, y)
            
    IOU_score, F1_score = save_evaluation(case)
    return IOU_score, F1_score

num_class = config['num_class']
model = net_20x_5x(num_classes=num_class)

checkpoint = torch.load(config['log_string_dir'] + config['best_weights'], map_location=f'cuda:{main_device}', weights_only=True)

model = nn.DataParallel(model, device_ids=device_ids)

try:
    model.load_state_dict(checkpoint['model_state_dict'])
except RuntimeError:
    new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model_state_dict'].items()}
    model.module.load_state_dict(new_state_dict)

device = torch.device(f"cuda:{main_device}" if torch.cuda.is_available() else "cpu")
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