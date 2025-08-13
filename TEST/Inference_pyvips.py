import os, sys
import torch
import numpy as np
from dataloader_pyvips import TumorDataModule
from model import net_20x_5x
from config_test import config
from tqdm import tqdm
import torch.nn as nn
from evaluation import *
import time
import pickle


def read_file_list(filename):
    file_list = []
    try:
        with open(filename, 'r') as f:
            for n in f:
                file_list.append(n.strip())
        return file_list
    except:
        print('[ERROR] Read file not found ' + filename)
        return []


def load_data_pkl(pkl_path):
    data = []
    with open(pkl_path, 'rb') as f:
        while True:
            try:
                data.append(pickle.load(f))
            except EOFError:
                break

    return data

class ClassifyModel():
    def __init__(self, config, case, cuda_number):
        super().__init__()

        self.config = config
        # setup logging
        self.cuda_number = cuda_number
        self.init_setting()
        self.case_name = case
        # load model and weights
        self.num_class = self.config['num_class']
        self.model = net_20x_5x(num_classes=self.num_class)

        self.model = nn.DataParallel(self.model, device_ids=[int(cuda_number)])
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        self.predict_dict = {}
            
    def data_loader(self,case_name):
        dataloader = dataloader = TumorDataModule(self.config, case=case_name, use_augmentation=True)
        self.test_dataloader = dataloader.test_dataloader()

    def init_setting(self):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda:" + self.cuda_number if torch.cuda.is_available() else "cpu")
        checkpoint_path = os.path.join(
            self.config['log_string_dir'], 
            self.config['best_weights']
        )
        self.checkpoint = torch.load(checkpoint_path)

    def forward_step(self, batch):
        img_high, img_low, x, y, label = batch
        img_high, img_low = img_high.to(self.device), img_low.to(self.device)
        with torch.amp.autocast('cuda'):
            prob = self.model(img_high, img_low)
        return prob, x, y

    def test(self):
        print("START INFERENCE {}".format(self.case_name) )
        init_para(self.case_name)
        # load dataset
        self.data_loader(self.case_name)

        softmax_layer = nn.Softmax(dim = 1)
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(self.test_dataloader)):
                prob, x, y = self.forward_step(batch)
                softmax_output = softmax_layer(prob)
                show_testing_img(prob, x, y)
                self.save_predict_info(batch, softmax_output, prob)

        IOU_score, F1_score = save_evaluation(self.case_name)
        print("IOU_score : {}".format(IOU_score))
        print("F1_score : {}".format(F1_score))
        print("INFERENCE {} DONE".format(self.case_name))
        
        return self.predict_dict, IOU_score, F1_score 
        
    def save_predict_info(self, batch, softmax_output, predict_output):
        img_high, img_low, x, y, label = batch
        _, pred = torch.max(predict_output, dim = 1)
        x = x.tolist()
        y = y.tolist()
        label = label.tolist()
        pred = pred.tolist()
        softmax_output = softmax_output.tolist()
        for i in range(len(label)):
            tile_name = self.case_name + "=" + str(x[i]) + "=" + str(y[i])
            self.predict_dict[tile_name] = [pred[i], label[i], softmax_output[i][0], softmax_output[i][1]]


# inference
if config['is_train'] == False:
#     case = "bf9fe238-54cd-4bc4-85bf-a6535424a519"
#     _round = 1
#     cuda_number = "7"
    _round = int(sys.argv[3])
    case = sys.argv[1]
    cuda_number = sys.argv[2]
    t_start = time.time()
    predict_dict, IOU_score, F1_score = ClassifyModel(config, case, cuda_number).test()
    t_end = time.time()
    print("Cost time: {} second".format(t_end-t_start))

tumor_past_choose_patch_list = read_file_list('choose_patch_name/tumor/'+case+'.txt')
normal_past_choose_patch_list = read_file_list('choose_patch_name/normal/'+case+'.txt')
tumor_past_error_list = read_file_list('patch_error/tumor/'+case+'.txt')
normal_past_error_list = read_file_list('patch_error/normal/'+case+'.txt')


confidence_dict = {}
label_dict = {}
pre_dict = {}
for patch in predict_dict:
    
    pre_dict[patch] = predict_dict[patch][0]
    label_dict[patch] = predict_dict[patch][1]
    confidence_dict[patch] = predict_dict[patch][3]

total_patch_count = len(pre_dict)
top_3percent_count = int(total_patch_count * 0.05)
sort_confidence_dict = sorted(confidence_dict.items(),key = lambda x:-x[1])

# +
tumor_predict_true = 0
normal_predict_true = 0

top_3_patch_png_file = []
top_3_patch_confidence = []
top_3_patch_error = []

low_3_patch_png_file = []
low_3_patch_confidence = []
low_3_patch_error = []

tumor_actually_count = 0
normal_actually_count = 0


tumor_index = 0
normal_index = -1

if _round < 11:
    
    f_t_e = open('patch_error/tumor/'+case+'.txt', 'a')
    while tumor_actually_count < top_3percent_count:
        patch_name = sort_confidence_dict[tumor_index][0]
        if confidence_dict[patch_name] < 0.65:
            break
        if patch_name in tumor_past_choose_patch_list:
            tumor_index += 1
            continue
        else:
            tumor_actually_count += 1
            top_3_patch_confidence.append(confidence_dict[patch_name])
            top_3_patch_png_file.append(patch_name)
            if label_dict[patch_name] == pre_dict[patch_name]:
                tumor_predict_true += 1
            else:
                f_t_e.write(patch_name)
                f_t_e.write('\n')
                top_3_patch_error.append(patch_name)
        tumor_index += 1
    f_t_e.close()

    
    f_n_e = open('patch_error/normal/'+case+'.txt', 'a')   
    while normal_actually_count < top_3percent_count:
        patch_name = sort_confidence_dict[normal_index][0]
        if confidence_dict[patch_name] > 0.35:
            break
        if patch_name in normal_past_choose_patch_list:
            normal_index -= 1
            continue
        else:
            normal_actually_count += 1
            low_3_patch_confidence.append(confidence_dict[patch_name])
            low_3_patch_png_file.append(patch_name)
            if label_dict[patch_name] == pre_dict[patch_name]:
                normal_predict_true += 1
            else:
                f_n_e.write(patch_name)
                f_n_e.write('\n')
                low_3_patch_error.append(patch_name)
        normal_index -= 1
    f_n_e.close()

    print('[INFO] tumor pseudo label accuracy:' + str(tumor_predict_true/tumor_actually_count))
    print('[INFO] normal pseudo label accuracy:' + str(normal_predict_true/normal_actually_count))
# -
if _round <= 11:
    path = 'output.txt'
    f = open(path, 'a')
    f.write('[INFO] case name:' + case)
    f.write('\n')
    f.write('[INFO] tumor choosing count:'+str(tumor_actually_count))
    f.write('\n')
    f.write('[INFO] noraml choosing count:'+str(normal_actually_count))
    f.write('\n')
    f.write('[INFO] tumor pseudo label accuracy:' + str(tumor_predict_true/tumor_actually_count))
    f.write('\n')
    f.write('[INFO] normal pseudo label accuracy:' + str(normal_predict_true/normal_actually_count))
    f.write('\n')
    f.write('[INFO] iou score:' + str(IOU_score))
    f.write('\n')
    f.write('[INFO] f1 score:'+str(F1_score))
    f.write('\n')
    f.write('----------------------------')
    f.write('\n')
    f.close()


def cut_unlabel_file_to_train(case, tumor_pl, normal_pl):
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BASE_DIR)
    PROJECT_NAME = os.path.basename(PROJECT_ROOT)
    
    pseudo_label_pkl_path = os.path.join(PROJECT_ROOT, 'PROCESSED_DATA', 'CASE_UUID', f'{PROJECT_NAME}_pseudo', f'{case}.pkl')
    wsi_path = os.path.join(PROJECT_ROOT, 'liver', 'tifs', f'{case}.tif')
    tumor_txt_path = os.path.join('choose_patch_name', 'tumor', f'{case}.txt')
    normal_txt_path = os.path.join('choose_patch_name', 'normal', f'{case}.txt')
    pl = []
    
    os.makedirs(os.path.dirname(pseudo_label_pkl_path), exist_ok=True)
    os.makedirs(os.path.dirname(tumor_txt_path), exist_ok=True)
    os.makedirs(os.path.dirname(normal_txt_path), exist_ok=True)
    
    f = open(tumor_txt_path, 'a')
    for tile_name in tumor_pl:
        x = tile_name.split("=")[1]
        y = tile_name.split("=")[2]
        pl.append([wsi_path, '0', str(x), str(y), "1"])
        f.write(tile_name)
        f.write('\n')
    f.close()
    
    f = open(normal_txt_path, 'a')
    for tile_name in normal_pl:
        x = tile_name.split("=")[1]
        y = tile_name.split("=")[2]
        pl.append([wsi_path, '0', str(x), str(y), "0"])
        f.write(tile_name)
        f.write('\n')
    f.close()
    
    new_pl = np.array(pl)
    
    if os.path.exists(pseudo_label_pkl_path) == False:
        print("no pseudo label pkl")
        with open(pseudo_label_pkl_path, 'wb') as w:
            pickle.dump(new_pl, w)
    else:
        print("exists pseudo label pkl")
        prev_pl = load_data_pkl(pseudo_label_pkl_path)
        mix_new_prev_pl = np.append(prev_pl[0], new_pl, axis = 0)
        with open(pseudo_label_pkl_path, 'wb') as w:
            pickle.dump(mix_new_prev_pl, w)


cut_unlabel_file_to_train(case, top_3_patch_png_file, low_3_patch_png_file)


