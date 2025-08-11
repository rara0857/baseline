import os

def read_file_list(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip()]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

download_list_path = os.path.join(BASE_DIR, 'download_list.txt')
UUIDs = read_file_list(download_list_path)

for case in UUIDs:
    print(f"[INFO] Processing {case}")
    case_dir = os.path.join(PROJECT_ROOT, 'PROCESSED_DATA', 'CASE_UUID', case)
    os.makedirs(case_dir, exist_ok=True)
    script_path = os.path.join(BASE_DIR, 'save_case_pkl.py')
    os.system(f"python {script_path} {case}")