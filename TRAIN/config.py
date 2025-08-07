import os
# set yout work root path
work_root_path = os.path.join('/work', os.listdir('/work')[0])

project = os.getcwd().split('/')[3]


def read_file_list(filename):
    file_list = []
    try:
        with open(filename, 'r') as f:
            for n in f:
                file_list.append(n.strip())
        return file_list
    except:
        print('[ERROR] Read file not found' + filename)

# +
train_list_path = '../train_list.txt'
val_list_path = '../val_list.txt'
unlabel_list_path = '../unlabel_list.txt'

config = {
    "data_pkl_path": '/work/rara0857/Baseline3/PROCESSED_DATA/CASE_UUID',
    "pseudo_label_path" : "/work/rara0857/Baseline3/PROCESSED_DATA/CASE_UUID/{}_pseudo".format(project),
    "stride_size":128,
    "patch_size":256,
    "train_batch_size" : 40,
    "val_batch_size" : 32,
    'unlabel_batch_size' : 32,
    "lr": 0.0001,
    "train_list": read_file_list(train_list_path),
    "val_list": read_file_list(val_list_path),
    "unlabel_list": read_file_list(unlabel_list_path),
    "num_class": 2,
    "max_iterator_num": 150001,
    "log_path": "LOG/logs/",
    "checkpoint_path":"best_weight.txt",
    "is_train": True,
    "re_train":False,
    "log_string_dir": "LOG/base40_lvl4_40x_10x_test/",
}