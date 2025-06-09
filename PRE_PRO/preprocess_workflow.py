import os
from multiprocessing import Pool


def read_file_list(filename):
    file_list = []
    try:
        with open(filename, 'r') as f:
            for n in f:
                file_list.append(n.strip())
        return file_list
    except:
        print('[ERROR] Read file not found' + filename)
UUID = read_file_list('download_list.txt')

for case in UUID:
    os.makedirs("/work/rara0857/Baseline3/PROCESSED_DATA/SAVE_PKL/" + case, exist_ok = True)
    os.system(f"python save_case_pkl.py {case}")
    print(f"python save_case_pkl.py {case}")


