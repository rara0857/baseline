import os

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

train_list_path = os.path.join(BASE_DIR, 'train_list.txt')
val_list_path = os.path.join(BASE_DIR, 'val_list.txt')

config = {
    "wsi_root_path": os.path.join(PROJECT_ROOT, "liver", "tifs"),
    "mask_root_path": os.path.join(PROJECT_ROOT, "liver", "masks"),
    "data_pkl_path": os.path.join(PROJECT_ROOT, "PROCESSED_DATA", "CASE_UUID"),
    "level": 0,
    "patch_size": 256,
    "stride_size": 128,
}