import os, sys
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


# set yout work root path
work_root_path = os.path.join('/work', os.listdir('/work')[0])
print("Your work root path is:", work_root_path)

# download annotation
annotation_work_path = f'{work_root_path}/DATA/RAW_DATA/USER_ANNOTATIONS/'

for case in UUID:
#     print(annotation_work_path+case)
#     print(annotation_work_path + case + '/annotation.json')
#     print('aws s3 cp ' + '--endpoint https://s3.twcc.ai s3://platform/image/data/'+ case+ '/annotation.json ')
    os.makedirs(f'{annotation_work_path}{case}', exist_ok=True)
    os.system('aws s3 cp ' + '--endpoint https://s3.twcc.ai s3://platform/image/data/'+ case+ '/annotation.json '+ annotation_work_path + case + '/annotation.json'  )

# download WSI(tif)
WSI_work_path = f'{work_root_path}/DATA/RAW_DATA/RAW_IMAGE/'
print(WSI_work_path)


def downloadtif(case, WSI_work_path=WSI_work_path):
    os.system(
        'aws s3 cp ' + '--endpoint https://s3.twcc.ai s3://platform/image/' + case + '.tif ' + WSI_work_path + case + '.tif')
    print(
        'aws s3 cp ' + '--endpoint https://s3.twcc.ai s3://platform/image/' + case + '.tif ' + WSI_work_path + case + '.tif')


os.makedirs(WSI_work_path, exist_ok=True)
POOL_SIZE = 4
pool = Pool(POOL_SIZE)
pool.map(downloadtif, UUID)
pool.close()
pool.join()
del pool



