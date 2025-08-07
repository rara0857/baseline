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
        try:
            weight_file='TRAIN/LOG/base40_lvl4_40x_10x_test/' + weight
            os.remove(weight_file)
        except:
            pass

def move_weight(round_):
    weight_path = 'TRAIN/LOG/base40_lvl4_40x_10x_test/best_model.pt'
    des_path = 'TRAIN/prev_logs/best_model_{}.pt'.format(round_)
    shutil.copyfile(weight_path, des_path)
    
def init_create_file():
    project = os.getcwd().split('/')[3]
    try:
        shutil.rmtree('TEST/choose_patch_name')
        shutil.rmtree('TEST/excel_error')
        shutil.rmtree('TEST/patch_error')
        shutil.rmtree('TEST/remove_patch')
        shutil.rmtree("/work/rara0857/Baseline3/PROCESSED_DATA/CASE_UUID/Baseline3_pseudo")
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
    os.makedirs('TEST/choose_patch_name/tumor', exist_ok = True)
    os.makedirs('TEST/choose_patch_name/normal', exist_ok = True)
    os.makedirs('TEST/excel_error/tumor', exist_ok = True)
    os.makedirs('TEST/excel_error/normal', exist_ok = True)
    os.makedirs('TEST/patch_error/tumor', exist_ok = True)
    os.makedirs('TEST/patch_error/normal', exist_ok = True)
    os.makedirs('TEST/remove_patch/tumor', exist_ok = True)
    os.makedirs('TEST/remove_patch/normal', exist_ok = True)
    os.makedirs('TRAIN/prev_logs', exist_ok = True)
    os.makedirs('/work/rara0857/Baseline3/PROCESSED_DATA/CASE_UUID/Baseline3_pseudo/', exist_ok=True)

def write_time(type_):
    f = open("time.txt", 'a')
    x = datetime.datetime.now()
    f.write(type_ + str(x))
    f.write('\n')
    f.close()
    
iter_number = [5000, 8000, 11000, 14000, 17000, 20000, 23000, 26000, 26000, 33000]
#iter_number = [1000, 1600, 2200, 2800, 3400, 4000, 4600, 5200, 5200, 6600]

n = 0
root = os.getcwd()

while n <= 9:
    
    print(n)
    
    if n <= 8 and n == 11:
        random.seed(5)
        
    if n == 0:
        print('[INFO] make init file')
        init_create_file()
        
    os.chdir(root)
    
    clear_weight()
                 
    os.chdir(os.path.join(root, 'TRAIN'))
    os.system('python main_execution.py ' + str(iter_number[n]))
    write_time("train")
    
    os.chdir(root)
    
    try:
        move_weight(n)
    except Exception  as e:
        print(e)
        pass
    
    if n != 9:
        os.chdir(os.path.join(root, 'TEST'))
        os.system('python pseudo_controller.py ' + str(n+1))
        write_time("test")
    
    if n % 2 == 0 and n != 0:
        os.system('python Back_out.py')

    n += 1
    
os.chdir(os.path.join(root, 'TEST'))
os.system('python test.py')
os.system('python draw_plot.py')