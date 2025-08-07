#!/usr/bin/env python
# coding: utf-8
# %%

import pandas as pd
import os
from multiprocessing import Pool
import multiprocessing
import numpy as np
import shutil
import pickle

# %%

class back_out():
    
    def __init__(self , file_name):
        
        self.file_name = file_name
        self.remove_patch(file_name)
    
    def read_file_list(self , filename):

        file_list=[]

        try:
            with open(filename, 'r') as f:
                for n in f:
                    file_list.append(n.strip())
            return file_list
        except:
            print('[ERROR] Read file not found ' + filename)
            return []
        
    def read_excel(self , file_name):

        tumor_excel = 'excel_error/tumor/tumor_Result_{}.csv'.format(file_name)
        tumor_excel = pd.read_csv(tumor_excel)
        tumor_excel = tumor_excel.drop(columns=['Unnamed: 0'])
        tumor_excel = tumor_excel.fillna(-1)

        normal_excel = 'excel_error/normal/normal_Result_{}.csv'.format(file_name)
        normal_excel = pd.read_csv(normal_excel)
        normal_excel = normal_excel.drop(columns=['Unnamed: 0'])
        normal_excel = normal_excel.fillna(-1)

        return tumor_excel , normal_excel 
    
    def reg(self , x , y):

        coefficients = np.polyfit(x,y,1) 
        p = np.poly1d(coefficients)
        return coefficients, p
    
    def remove_patch(self , file_name):
        pseudo_path = "/work/rara0857/Baseline3/PROCESSED_DATA/CASE_UUID/Baseline3_pseudo/{}.pkl".format(file_name)
        wsi_path = '/work/rara0857/Baseline3/liver/tifs/{}.tif'.format(self.file_name)
        print(pseudo_path)
        print(wsi_path)
        tumor_excel , normal_excel =  self.read_excel(file_name)
        
        remove_patch_tumor_txt = 'remove_patch/tumor/{}.txt'.format(file_name)
        remove_patch_normal_txt = 'remove_patch/normal/{}.txt'.format(file_name)
        prev_remove_patch_tumor = self.read_file_list(remove_patch_tumor_txt)
        prev_remove_patch_normal = self.read_file_list(remove_patch_normal_txt)
        f_t = open(remove_patch_tumor_txt,'a')
        f_n = open(remove_patch_normal_txt,'a')
        keep_list = []
        
        for index in range(tumor_excel.shape[0]):
            
            patch_name = tumor_excel.iloc[index][0]
            tmp = []
            x = patch_name.split("=")[1]
            y = patch_name.split("=")[2]
            for row in range(1,tumor_excel.shape[1]):
                if tumor_excel.iloc[index][row]!=-1:
                    tmp.append(tumor_excel.iloc[index][row])
                    

            each_confidence = np.array(tmp)
            var = np.std(each_confidence)
            if var >= 0.15:
                axis_x = np.arange(len(each_confidence))
                axis_y = np.array(each_confidence)
                (arg1, arg2), text1 = self.reg(axis_x, axis_y) 
                if arg1 > 0:
                    keep_list.append([wsi_path, "0", x, y, "1"])
                else:
                    if patch_name not in prev_remove_patch_tumor:
                        print(patch_name)
                        f_t.write(patch_name)
                        f_t.write('\n')
            else:
                keep_list.append([wsi_path, "0", x, y, "1"])
        f_t.close()
        
        for index in range(normal_excel.shape[0]):

            patch_name = normal_excel.iloc[index][0]
            tmp = []
            x = patch_name.split("=")[1]
            y = patch_name.split("=")[2]
            for row in range(1,normal_excel.shape[1]):
                if normal_excel.iloc[index][row] != -1:
                    tmp.append(normal_excel.iloc[index][row])
               
                
            each_confidence = np.array(tmp)
            var = np.std(each_confidence)
            if var >= 0.15:
                axis_x = np.arange(len(each_confidence))
                axis_y = np.array(each_confidence)
                (arg1, arg2), text1 = self.reg(axis_x,axis_y) 
                if arg1 < 0:
                     keep_list.append([wsi_path, "0", x, y, "0"])
                else:
                    if patch_name not in prev_remove_patch_normal:
                        print(patch_name)
                        f_n.write(patch_name)
                        f_n.write('\n')
            else:
                 keep_list.append([wsi_path, "0", x, y, "0"])
        f_n.close()
        
        print(len(keep_list))
        keep_list = np.array(keep_list)
        os.makedirs(os.path.dirname(pseudo_path), exist_ok=True)
        with open(pseudo_path, 'wb') as w:
            pickle.dump(keep_list, w)


# %%


if __name__ == '__main__':

    case = []
    for file_name in os.listdir('excel_error/tumor/'):
        case.append(file_name.split('_')[-1].split('.')[0])
        
    POOL_SIZE = 32
    pool = Pool(POOL_SIZE)
    pool.map(back_out, case)
    pool.close()
    pool.join()
    del pool


# %%

# %%
