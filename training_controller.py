#!/usr/bin/env python
# coding: utf-8
import os
import shutil
import random
import datetime

def read_file_list(filename):
    file_list = []
    try:
        with open(filename, 'r') as f:
            for n in f:
                file_list.append(n.strip())
        return file_list
    except:
        print('[ERROR] Read file not found' + filename)
        return []

def clear_weight():
    for weight in os.listdir('TRAIN/LOG/base40_lvl4_40x_10x_test'):
        if weight.startswith('model.ckpt-') and weight.endswith('.pt'):
            try:
                weight_file = os.path.join('TRAIN', 'LOG', 'base40_lvl4_40x_10x_test', weight)
                os.remove(weight_file)
            except:
                pass

def move_weight(round_):
    weight_path = os.path.join('TRAIN', 'LOG', 'base40_lvl4_40x_10x_test', 'best_model.pt')
    des_path = os.path.join('TRAIN', 'prev_logs', f'best_model_{round_}.pt')
    shutil.copyfile(weight_path, des_path)

def clean_processed_data():
    processed_data_path = os.path.join('PROCESSED_DATA', 'CASE_UUID')
    
    if not os.path.exists(processed_data_path):
        return
    
    for case_folder in os.listdir(processed_data_path):
        case_folder_path = os.path.join(processed_data_path, case_folder)
        if os.path.isdir(case_folder_path):
            for filename in os.listdir(case_folder_path):
                file_path = os.path.join(case_folder_path, filename)
                if not filename.endswith('.pkl'):
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        pass
    
def init_create_file():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_NAME = os.path.basename(BASE_DIR)
    
    try:
        shutil.rmtree('TEST/choose_patch_name')
        shutil.rmtree('TEST/excel_error')
        shutil.rmtree('TEST/patch_error')
        shutil.rmtree('TEST/remove_patch')
        shutil.rmtree(os.path.join('PROCESSED_DATA', 'CASE_UUID', f'{PROJECT_NAME}_pseudo'))
        shutil.rmtree('TRAIN/prev_logs')         
    except:
        pass
    try:
        os.remove("TEST/output.txt")
    except:
        pass
    try:
        os.remove("TRAIN/aug.npy")
    except:
        pass
    
    os.makedirs('TEST/choose_patch_name/tumor', exist_ok=True)
    os.makedirs('TEST/choose_patch_name/normal', exist_ok=True)
    os.makedirs('TEST/excel_error/tumor', exist_ok=True)
    os.makedirs('TEST/excel_error/normal', exist_ok=True)
    os.makedirs('TEST/patch_error/tumor', exist_ok=True)
    os.makedirs('TEST/patch_error/normal', exist_ok=True)
    os.makedirs('TEST/remove_patch/tumor', exist_ok=True)
    os.makedirs('TEST/remove_patch/normal', exist_ok=True)
    os.makedirs('TRAIN/prev_logs', exist_ok=True)
    os.makedirs(os.path.join('PROCESSED_DATA', 'CASE_UUID', f'{PROJECT_NAME}_pseudo'), exist_ok=True)

def write_time(type_):
    with open("time.txt", 'a') as f:
        x = datetime.datetime.now()
        f.write(type_ + str(x) + '\n')
    
iter_number = [5000, 8000, 11000, 14000, 17000, 20000, 23000, 26000, 26000, 33000]
n = 0
root = os.getcwd()
iterations = len(iter_number)

while n < iterations:
    print(f"Round {n+1}/{iterations}: {iter_number[n]} iterations")
    
    if n <= 8 and n == 11:
        random.seed(5)
        
    if n == 0:
        print('[INFO] make init file')
        init_create_file()
        clean_processed_data()
        
    os.chdir(root)
    
    if n > 0:
        clear_weight()
                 
    os.chdir(os.path.join(root, 'TRAIN'))
    result = os.system('python main_execution.py ' + str(iter_number[n]))
    write_time("train")
    
    os.chdir(root)
    
    try:
        move_weight(n)
    except Exception as e:
        print(e)
        pass

    if n != iterations-1:
        os.chdir(os.path.join(root, 'TEST'))
        inference_result = os.system('python pseudo_controller.py ' + str(n+1))
        write_time("test")
    
    if n % 2 == 0 and n != 0:
        os.system('python Back_out.py')

    n += 1
    
os.chdir(os.path.join(root, 'TEST'))
os.system('python test.py')
os.system('python draw_plot.py')