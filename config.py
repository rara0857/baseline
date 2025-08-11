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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

config = {
    "preprocess_save_path": os.path.join(BASE_DIR, "PROCESSED_DATA", "CASE_UUID"),
    "wsi_root_path": os.path.join(BASE_DIR, "liver", "tifs"),
    "data_pkl_path": os.path.join(BASE_DIR, "PROCESSED_DATA", "CASE_UUID"),
    "level": 0,
    "patch_size": 256,
    "stride_size": 128,
    "train_list": read_file_list(os.path.join(BASE_DIR, train_list_path)),
    "val_list": read_file_list(os.path.join(BASE_DIR, val_list_path)),
}
# -

