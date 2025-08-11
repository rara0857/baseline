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

test_list_path = os.path.join(PROJECT_ROOT, 'test_list.txt')

config = {
    "wsi_root_path": os.path.join(PROJECT_ROOT, 'liver', 'tifs'),
    "mask_path": os.path.join(PROJECT_ROOT, 'liver', 'masks'),
    "data_pkl_path": os.path.join(PROJECT_ROOT, 'PROCESSED_DATA', 'CASE_UUID'),
    "patch_size": 256,
    "stride_size": 128,
    "test_batch_size": 64,
    "test_list": read_file_list(test_list_path),
    "num_class": 2,
    "is_train": False,  # test
    "model_checkpoint_path": "",
    "log_string_dir": os.path.join(PROJECT_ROOT, "TRAIN", "LOG", "base40_lvl4_40x_10x_test"),
    "best_weights": "best_model.pt", 
    "threshold": [0.65, 0.6, 0.55, 0.5]
}