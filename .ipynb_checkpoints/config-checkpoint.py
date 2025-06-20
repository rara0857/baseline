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

# +
train_list_path = 'train_list.txt'
val_list_path = 'val_list.txt'
# set yout work root path
work_root_path = os.path.join('/work', os.listdir('/work')[0])

config = {
    "preprocess_save_path":f'{work_root_path}/DATA/PROCESSED_DATA/CASE_UUID',
    "annotation_json_path":f'{work_root_path}/DATA/RAW_DATA/USER_ANNOTATIONS',
    "wsi_root_path":f'{work_root_path}/DATA/RAW_DATA/RAW_IMAGE',
    "data_pkl_path":f'{work_root_path}/DATA/PROCESSED_DATA/CASE_UUID/',
    "level":0,
    "patch_size" :256,
    "stride_size":128,
    "train_list": read_file_list(train_list_path),
    "val_list": read_file_list(val_list_path),
}

# -

