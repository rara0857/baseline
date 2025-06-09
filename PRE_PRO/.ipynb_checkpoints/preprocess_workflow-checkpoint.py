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

# +
# # Download tif and annotation json
# os.system("python Download_Data.py")

# +
# # # json
# def json2xml(case):
#     os.system("python json2xml.py "+case)

# POOL_SIZE = 1
# pool = Pool(POOL_SIZE)
# pool.map(json2xml, UUID)
# pool.close()
# pool.join()
# del pool

# +
# # # xml to mask
# def xml2mask(case):
#     os.system("python Gen_mask.py "+ case)

# POOL_SIZE = 4
# pool = Pool(POOL_SIZE)
# pool.map(xml2mask, UUID)
# pool.close()
# pool.join()
# del pool
# -

# save patch position
for case in UUID:
    os.makedirs("/work/jack91630/DATA/PROCESSED_DATA/SAVE_PKL/" + case, exist_ok = True)
    os.system(f"python save_case_pkl.py {case}")
    print(f"python save_case_pkl.py {case}")


