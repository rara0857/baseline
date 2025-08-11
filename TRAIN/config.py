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
PROJECT_NAME = os.path.basename(PROJECT_ROOT)

# +
train_list_path = os.path.join(PROJECT_ROOT, 'train_list.txt')
val_list_path = os.path.join(PROJECT_ROOT, 'val_list.txt')
unlabel_list_path = os.path.join(PROJECT_ROOT, 'unlabel_list.txt')

config = {
    "data_pkl_path": os.path.join(PROJECT_ROOT, 'PROCESSED_DATA', 'CASE_UUID'),
    "pseudo_label_path": os.path.join(PROJECT_ROOT, 'PROCESSED_DATA', 'CASE_UUID', f"{PROJECT_NAME}_pseudo"),
    "stride_size": 128,
    "patch_size": 256,
    "train_batch_size": 40,
    "val_batch_size": 32,
    'unlabel_batch_size': 32,
    "lr": 0.0001,
    "train_list": read_file_list(train_list_path),
    "val_list": read_file_list(val_list_path),
    "unlabel_list": read_file_list(unlabel_list_path),
    "num_class": 2,
    "max_iterator_num": 150001,
    "log_path": os.path.join(BASE_DIR, "LOG", "logs"),
    "checkpoint_path": "best_weight.txt",
    "is_train": True,
    "re_train": False,
    "log_string_dir": os.path.join(BASE_DIR, "LOG", "base40_lvl4_40x_10x_test"),
}