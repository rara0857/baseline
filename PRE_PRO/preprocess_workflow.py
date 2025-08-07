import os

def read_file_list(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip()]

UUIDs = read_file_list(f'/work/rara0857/Baseline3/PRE_PRO/download_list.txt')

for case in UUIDs:
    print(f"[INFO] Processing {case}")
    os.makedirs(f"/work/rara0857/Baseline3/PROCESSED_DATA/CASE_UUID/{case}", exist_ok=True)
    os.system(f"python PRE_PRO/save_case_pkl.py {case}")
