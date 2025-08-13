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
        try:
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
        except RuntimeError as e:
            new_state_dict = {k.replace("module.", ""): v for k, v in self.checkpoint['model_state_dict'].items()}
            self.model.module.load_state_dict(new_state_dict)
        self.model.to(self.device)
        
        self.predict_dict_1 = {}
        self.predict_dict_2 = {}
        self.predict_dict_3 = {}
            
    def data_loader(self,case_name):
        dataloader = TumorDataModule(self.config, case=case_name, use_augmentation=True)
        self.test_dataloader = dataloader.test_dataloader()

    def init_setting(self):
        self.device = torch.device("cuda:" + self.cuda_number if torch.cuda.is_available() else "cpu")
        checkpoint_path = os.path.join(
            self.config['log_string_dir'], 
            self.config['best_weights']
        )
        self.checkpoint = torch.load(
            checkpoint_path,
            map_location=f'cuda:{self.cuda_number}',
            weights_only=True
        )

    def forward_step(self, batch):
        img_high_1, img_low_1, img_high_2, img_low_2, img_high_3, img_low_3, x, y, label = batch
        img_high_1, img_low_1 = img_high_1.to(self.device), img_low_1.to(self.device)
        img_high_2, img_low_2 = img_high_2.to(self.device), img_low_2.to(self.device)
        img_high_3, img_low_3 = img_high_3.to(self.device), img_low_3.to(self.device)
        with torch.amp.autocast('cuda'):
            prob_1 = self.model(img_high_1, img_low_1)
            prob_2 = self.model(img_high_2, img_low_2)
            prob_3 = self.model(img_high_3, img_low_3)
        return prob_1, prob_2, prob_3, x, y

    def test(self):
        print("START INFERENCE {}".format(self.case_name) )
        init_para(self.case_name)
        # load dataset
        self.data_loader(self.case_name)

        softmax_layer = nn.Softmax(dim = 1)
        self.model.eval()
        
        #====================================== dropout-activate ================================================        
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                print("[INFO] drop out layer activate")
                m.train()
        #====================================== dropout-activate ================================================  
        
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(self.test_dataloader)):
                
                prob_1, prob_2, prob_3,x, y = self.forward_step(batch)
                softmax_output_1 = softmax_layer(prob_1)
                softmax_output_2 = softmax_layer(prob_2)
                softmax_output_3 = softmax_layer(prob_3)
                
                show_testing_img(prob_1, x, y)
                self.save_predict_info(batch, softmax_output_1, softmax_output_2, softmax_output_3, prob_1, prob_2, prob_3)

        IOU_score, F1_score = save_evaluation(self.case_name)
        print("IOU_score : {}".format(IOU_score))
        print("F1_score : {}".format(F1_score))
        print("INFERENCE {} DONE".format(self.case_name))
        
        return self.predict_dict_1, self.predict_dict_2, self.predict_dict_3, IOU_score, F1_score 
        
    def save_predict_info(self, batch, softmax_output_1, softmax_output_2, softmax_output_3, predict_output_1, predict_output_2, predict_output_3):
        img_high_1, img_low_1, img_high_2, img_low_2, img_high_3, img_low_3, x, y, label = batch
        _, pred_1 = torch.max(predict_output_1, dim = 1)
        _, pred_2 = torch.max(predict_output_2, dim = 1)
        _, pred_3 = torch.max(predict_output_3, dim = 1)
        x = x.tolist()
        y = y.tolist()
        label = label.tolist()
        pred_1 = pred_1.tolist()
        pred_2 = pred_2.tolist()
        pred_3 = pred_3.tolist()
        softmax_output_1 = softmax_output_1.tolist()
        softmax_output_2 = softmax_output_2.tolist()
        softmax_output_3 = softmax_output_3.tolist()
        for i in range(len(label)):
            tile_name = self.case_name + "=" + str(x[i]) + "=" + str(y[i])
            self.predict_dict_1[tile_name] = [pred_1[i], label[i], softmax_output_1[i][0], softmax_output_1[i][1]]
            self.predict_dict_2[tile_name] = [pred_2[i], label[i], softmax_output_2[i][0], softmax_output_2[i][1]]
            self.predict_dict_3[tile_name] = [pred_3[i], label[i], softmax_output_3[i][0], softmax_output_3[i][1]]


# inference
if config['is_train'] == False:
    _round = int(sys.argv[3])
    case = sys.argv[1]
    cuda_number = sys.argv[2]
    t_start = time.time()
    predict_dict_1, predict_dict_2, predict_dict_3, IOU_score, F1_score = ClassifyModel(config, case, cuda_number).test()
    t_end = time.time()
    print("Cost time: {} second".format(t_end-t_start))

tumor_past_choose_patch_list = read_file_list('choose_patch_name/tumor/'+case+'.txt')
normal_past_choose_patch_list = read_file_list('choose_patch_name/normal/'+case+'.txt')
tumor_past_error_list = read_file_list('patch_error/tumor/'+case+'.txt')
normal_past_error_list = read_file_list('patch_error/normal/'+case+'.txt')


# +
confidence_dict_1 = {}
label_dict_1 = {}
pre_dict_1 = {}

confidence_dict_2 = {}
label_dict_2 = {}
pre_dict_2 = {}

confidence_dict_3 = {}
label_dict_3 = {}
pre_dict_3 = {}

for patch in predict_dict_1:

    pre_dict_1[patch] = predict_dict_1[patch][0]
    label_dict_1[patch] = predict_dict_2[patch][1]
    confidence_dict_1[patch] = predict_dict_3[patch][3]
    
    if _round <= 7:
        
        pre_dict_2[patch] = predict_dict_2[patch][0]
        label_dict_2[patch] = predict_dict_2[patch][1]
        confidence_dict_2[patch] = predict_dict_2[patch][3]

        pre_dict_3[patch] = predict_dict_3[patch][0]
        label_dict_3[patch] = predict_dict_3[patch][1]
        confidence_dict_3[patch] = predict_dict_3[patch][3]

# +
var_dict = {}

if _round <= 7:

    for patch in predict_dict_1:
        var = np.std([confidence_dict_1[patch],confidence_dict_2[patch],confidence_dict_3[patch]])
        var_dict[patch] = var
# -

if _round <= 7:

    total_patch_count = len(pre_dict_1)
    top_3percent_count = int(total_patch_count * 0.05)
    sort_confidence_dict = sorted(confidence_dict_1.items(), key = lambda x:-x[1])

    try:
        print("[INFO] confidence_threshold: {}".format(config["threshold"][max(0, (_round - 1) // 3)]))
        confidence_threshold = config["threshold"][max(0, (_round - 1) // 3)]
    except:
        print("bug bug bug bug bug")
        pass

# +
top_3_patch_png_file = []
top_3_patch_confidence = []

low_3_patch_png_file = []
low_3_patch_confidence = []

uncertainty_tumor_predict_true = 0
uncertainty_normal_predict_true = 0

uncertainty_tumor_choose_count = 0
uncertainty_normal_choose_count = 0

tumor_index = 0
normal_index = -1

if _round <= 7:

    f_t_e = open('patch_error/tumor/' + case + '.txt', 'a')
    while uncertainty_tumor_choose_count < top_3percent_count:
        patch_name = sort_confidence_dict[tumor_index][0]
        patch_confidence = sort_confidence_dict[tumor_index][1]
        if patch_confidence < confidence_threshold:
            break
        if patch_name in tumor_past_choose_patch_list:
            tumor_index += 1
            continue
        if var_dict[patch_name] < 0.2:
            top_3_patch_png_file.append(patch_name)
            top_3_patch_confidence.append(confidence_dict_1[patch_name])
            uncertainty_tumor_choose_count += 1
            if label_dict_1[patch_name] == pre_dict_1[patch_name]:
                uncertainty_tumor_predict_true += 1
            else:
                f_t_e.write(patch_name)
                f_t_e.write('\n')
        tumor_index += 1
    f_t_e.close()


    f_n_e = open('patch_error/normal/' + case + '.txt', 'a')              
    while uncertainty_normal_choose_count < top_3percent_count:
        patch_name = sort_confidence_dict[normal_index][0]
        patch_confidence = sort_confidence_dict[normal_index][1]
        if patch_confidence > (1 - confidence_threshold):
            break
        if patch_name in normal_past_choose_patch_list:
            normal_index -= 1
            continue
        if var_dict[patch_name] < 0.2:
            low_3_patch_png_file.append(patch_name)
            low_3_patch_confidence.append(confidence_dict_1[patch_name])
            uncertainty_normal_choose_count += 1
            if label_dict_1[patch_name] == pre_dict_1[patch_name]:
                uncertainty_normal_predict_true += 1
            else:
                f_n_e.write(patch_name)
                f_n_e.write('\n')
        normal_index -= 1
    f_n_e.close()


    if len(top_3_patch_png_file) > 0:
        print('[INFO] tumor pseudo label accuracy:' + str(uncertainty_tumor_predict_true / len(top_3_patch_png_file)))
    else:
        print('[INFO] tumor pseudo label accuracy: No samples selected')
    
    if len(low_3_patch_png_file) > 0:
        print('[INFO] normal pseudo label accuracy:' + str(uncertainty_normal_predict_true / len(low_3_patch_png_file)))
    else:
        print('[INFO] normal pseudo label accuracy: No samples selected')
# -
if _round <= 7:
    path = 'output.txt'
    f = open(path, 'a')
    f.write('[INFO] case name:' + case)
    f.write('\n')
    f.write('[INFO] tumor choosing count:' + str(len(top_3_patch_png_file)))
    f.write('\n')
    f.write('[INFO] noraml choosing count:' + str(len(low_3_patch_png_file)))
    f.write('\n')
    if len(top_3_patch_png_file) > 0:
        f.write('[INFO] tumor pseudo label accuracy:' + str(uncertainty_tumor_predict_true / len(top_3_patch_png_file)))
    else:
        f.write('[INFO] tumor pseudo label accuracy: No samples selected')
    f.write('\n')
    if len(low_3_patch_png_file) > 0:
        f.write('[INFO] normal pseudo label accuracy:' + str(uncertainty_normal_predict_true / len(low_3_patch_png_file)))
    else:
        f.write('[INFO] normal pseudo label accuracy: No samples selected')
    f.write('\n')
    f.write('[INFO] iou score:' + str(IOU_score))
    f.write('\n')
    f.write('[INFO] f1 score:' + str(F1_score))
    f.write('\n')
    f.write('----------------------------')
    f.write('\n')
    f.close()

# +
import pandas as pd

tumor_res = 0
normal_res = 0
tumor_past_error_list = read_file_list('patch_error/tumor/' + case + '.txt')
normal_past_error_list = read_file_list('patch_error/normal/' + case + '.txt')


if _round == 1:
    print(_round)
    d = {'Index' : top_3_patch_png_file,'iter_' + str(_round) : top_3_patch_confidence}
    tumor_res = pd.DataFrame(d)
    
else:
    print(_round)
    if _round <= 7:
    
        d = {'Index' : top_3_patch_png_file,'iter_'+str(_round):top_3_patch_confidence}
        tumor_res = pd.DataFrame(d)

        last = pd.read_csv('excel_error/tumor/tumor_Result_' + case + '.csv')
        last = last.drop(columns = ['Unnamed: 0'])
        last_shape = last.shape[0]
        tumor_res = pd.concat([last,tumor_res],axis = 0,ignore_index = True)
        
    else:
        
        last = pd.read_csv('excel_error/tumor/tumor_Result_' + case + '.csv')
        last = last.drop(columns = ['Unnamed: 0'])
        last_shape = last.shape[0]
        tumor_res = last
        
    for index in range(last_shape):
        patch_name = tumor_res.iloc[index][0]
        tumor_res.loc[tumor_res['Index'] == patch_name,'iter_' + str(_round)] = confidence_dict_1[patch_name]


    
    
if _round == 1:
    print(_round)
    d = {'Index' : low_3_patch_png_file,'iter_' + str(_round) : low_3_patch_confidence}
    normal_res = pd.DataFrame(d)
    
else:
    print(_round)
    if _round <= 7:
        d = {'Index':low_3_patch_png_file,'iter_' + str(_round) : low_3_patch_confidence}
        normal_res = pd.DataFrame(d)

        last = pd.read_csv('excel_error/normal/normal_Result_' + case + '.csv')
        last = last.drop(columns = ['Unnamed: 0'])
        last_shape = last.shape[0]
        normal_res = pd.concat([last,normal_res],axis = 0, ignore_index = True)
        
    else:
        
        last = pd.read_csv('excel_error/normal/normal_Result_' + case + '.csv')
        last = last.drop(columns = ['Unnamed: 0'])
        last_shape = last.shape[0]
        normal_res = last
        
    for index in range(last_shape):
        patch_name = normal_res.iloc[index][0]
        normal_res.loc[normal_res['Index'] == patch_name,'iter_'+str(_round)] = confidence_dict_1[patch_name]
# -

tumor_res.to_csv('excel_error/tumor/tumor_Result_' + case + '.csv')
normal_res.to_csv('excel_error/normal/normal_Result_' + case + '.csv')


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
    
    with open(tumor_txt_path, 'a') as f:
        for tile_name in tumor_pl:
            x = tile_name.split("=")[1]
            y = tile_name.split("=")[2]
            pl.append([wsi_path, '0', str(x), str(y), "1"])
            f.write(tile_name + '\n')
    
    with open(normal_txt_path, 'a') as f:
        for tile_name in normal_pl:
            x = tile_name.split("=")[1]
            y = tile_name.split("=")[2]
            pl.append([wsi_path, '0', str(x), str(y), "0"])
            f.write(tile_name + '\n')
    
    new_pl = np.array(pl)
    if not os.path.exists(pseudo_label_pkl_path):
        print("no pseudo label pkl")
        with open(pseudo_label_pkl_path, 'wb') as w:
            pickle.dump(new_pl, w)
    else:
        print("exists pseudo label pkl")
        prev_pl = load_data_pkl(pseudo_label_pkl_path)
        mix_new_prev_pl = np.append(prev_pl[0], new_pl, axis=0)
        with open(pseudo_label_pkl_path, 'wb') as w:
            pickle.dump(mix_new_prev_pl, w)

if _round <= 7:

    cut_unlabel_file_to_train(case, top_3_patch_png_file, low_3_patch_png_file)