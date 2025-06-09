import sys, os, math
#import data reader API
# sys.path.append('/opt/ASAP/bin')
import openslide
# import multiresolutionimageinterface as mir
from multiprocessing import Process
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import cv2
import time

patch_threshold = 205
total_process_num = 0
finished_process_num = 0
produce_patches_process_list = []

# geometric_series_sum(1.0, 2.0, float(0))
def geometric_series_sum(a, r, n):
    return a * (1.0 - pow(r, n)) / (1.0 - r)

def init_reader(image_path, type='none'):
    if os.path.isfile(image_path):
        print('init_reader: ' + image_path)
    else:
        print('init_reader: No such file.')
        return None

    mr_data = openslide.OpenSlide(image_path)

    return mr_data

def get_gt_img(mr_data , stride_size=128):
    width = int(mr_data.dimensions[0] / stride_size) + 1
    length = int(mr_data.dimensions[1] / stride_size) + 1
    # print(width,length)
    # print(mr_data.getDimensions()[0],mr_data.getDimensions()[1])

    data_gt = mr_data.read_region((0,0),level=int(math.log2(stride_size)),size=(width, length))
    # data_gt = mr_data.getUCharPatch(0, 0, width, length, int(math.log2(stride_size)))
    
    data_gt = np.squeeze(data_gt,axis=2)
    data_gt = np.uint8(data_gt)

    return data_gt

def get_total_patch(mr_data, stride_size, threshold):
    '''
    using thumbnail(1 stride size patch represent 1 pixel) to identify the
    foreground patches
    '''
    width = int(mr_data.dimensions[0] / stride_size) + 1
    length = int(mr_data.dimensions[1] / stride_size) + 1

    data_patch = mr_data.read_region((0,0),level=int(math.log2(stride_size)),size=(width, length))
    # data_patch = mr_data.getUCharPatch(0, 0, width, length, int(math.log2(stride_size)))
    data_patch = np.uint8(data_patch)
    np.seterr(divide='ignore', invalid='ignore')
    sat_t = np.nan_to_num(1-np.amin(data_patch,axis=2)/np.amax(data_patch,axis=2))

    return (sat_t >= 0.1).sum()#(data_patch.mean(axis=2) <= threshold).sum()

def transform(image):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    image= test_transform(image)
    return image

def read_patch(mr_data, level, x_tile_stride, y_tile_stride, patch_size=256):
    if level == 0:
        offset = 0
        x = x_tile_stride - offset
        y = y_tile_stride - offset
        data_patch = mr_data.read_region((int(x), int(y)),level=level,size=(patch_size, patch_size))
        data_patch = np.uint8(data_patch)
        data_patch = data_patch[:, :, :3]
        return data_patch

    elif level == 1:
        offset = (patch_size / 2) * geometric_series_sum(1.0, 2.0, float(level+1))

        x = x_tile_stride - offset
        y = y_tile_stride - offset
        
        data_patch = mr_data.read_region((int(x), int(y)),level=level,size=(patch_size, patch_size))
        data_patch = np.uint8(data_patch)
        data_patch = data_patch[:, :, :3]

        return data_patch


def get_level_patches_list(mr_data, level_set, x_tile_stride, y_tile_stride, patch_size=256):
    # get multiple magnification patches with overlapping
    level_patch_dict = dict()

    offset = (patch_size / 2) * geometric_series_sum(1.0, 2.0, float(0))
    x = x_tile_stride - offset
    y = y_tile_stride - offset
    tile_name = str(int(x)) + '-' + str(int(y))

    level_patch_dict[0] = read_patch(mr_data=mr_data,
                                     level=0,
                                     x_tile_stride=x_tile_stride,
                                     y_tile_stride=y_tile_stride,
                                     patch_size=patch_size)


    # remove background
    detect_region = level_patch_dict[0][64:192, 64:192]
    np.seterr(divide='ignore', invalid='ignore')
    sat = np.nan_to_num(1-np.amin(detect_region,axis=2)/np.amax(detect_region,axis=2)) # definition of saturation, in pixel-wise, axis=2: [R G B]
    pix_sat_count = (sat < 0.1).sum() # saturation threshold
    all_pix_count = (sat > -1).sum() # get all pixels
    
    if pix_sat_count > all_pix_count*0.75: #detect_region.mean() > patch_threshold:
        return False, level_patch_dict, tile_name
    
    for i in level_set:
        if i > 0:
            level_patch_dict[i] = read_patch(mr_data=mr_data,
                                             level=i,
                                             x_tile_stride=x_tile_stride,
                                             y_tile_stride=y_tile_stride,
                                             patch_size=patch_size)

        # cv2.imwrite(str(i) +"_"+str(tile_name)+ '_result.png', level_patch_dict[i])
        level_patch_dict[i]= Image.fromarray(level_patch_dict[i])
#         print(np.shape(level_patch_dict[i]),"#(256, 256, 3)")
        level_patch_dict[i] = transform(level_patch_dict[i])


    #         print(i,type(level_patch_dict[i]))
        
    return True, level_patch_dict, tile_name


def read_rect_patches(image_path, start_loc, end_loc, level_set, patches_queue, lock, batch_size=8, patch_size=256, stride_size=128):
    # divide patches
    batch_size_counter = 0
    batch_list = []
    
    mr_data = init_reader(image_path, type='none')
    
    if mr_data == None:
        print('read_rect_patches: init_reader failure.')
        return None
    
    for x_tile_stride in range(start_loc, end_loc, stride_size):
        for y_tile_stride in range(0, int(mr_data.dimensions[1]), stride_size):
            '''
                read multi-magnification patches
            '''
            # level_patch_dict = [True, level_patch_dict, tile_name]
            level_patch_dict = get_level_patches_list(mr_data=mr_data,
                                                      level_set=level_set,
                                                      x_tile_stride=x_tile_stride,
                                                      y_tile_stride=y_tile_stride,
                                                      patch_size=patch_size)

            if level_patch_dict[0]: # only pass foreground patch
                batch_list.append(level_patch_dict)
                batch_size_counter += 1

                if batch_size_counter == batch_size:
                    lock.acquire()
                    patches_queue.put(batch_list)
                    lock.release()
                    batch_size_counter = 0
                    batch_list = []
                    
    lock.acquire()
    patches_queue.put('DONE')
    print("PUT DONE")
    lock.release()

def read_batch_patches(image_path, level_set, patches_queue, lock, process_num, batch_size=8, patch_size=256, stride_size=128):
    
    mr_data = init_reader(image_path, type='none')

    '''
        init batch config
    '''
    global produce_patches_process_list, total_process_num
    
    total_process_num = process_num
    rect_stride_size = int(mr_data.dimensions[0] / process_num) # the region that one process works on
    extra_stride = mr_data.dimensions[0] % process_num
    rect_offset = rect_stride_size
    # total patch num indicate the numbers of patch that contains object
    total_patch_num = get_total_patch(mr_data=mr_data, stride_size=stride_size, threshold=patch_threshold)

    for rect_stride in range(0, int(mr_data.dimensions[0]), rect_stride_size):        
        '''
            read multi-magnification patches
        '''
        # dealing with the extra region could not be divided
        if rect_stride == (rect_stride_size * process_num) and extra_stride > 0:
            rect_offset = extra_stride
            process_num+=1 # add one for extra needed process
            total_process_num = process_num
        print(rect_stride + rect_offset)
        
        '''
            Fork new prrocess.
        '''
        # read patched for each region that works on each process
        produce_patches_process = Process(target=read_rect_patches,
                                          args=(image_path, rect_stride, rect_stride + rect_offset, level_set, patches_queue, lock, batch_size, patch_size, stride_size))

                        # def read_rect_patches(image_path, start_loc, end_loc, level_set, patches_queue, lock, batch_size=8,patch_size=256, stride_size=128):
        produce_patches_process_list.append(produce_patches_process)
        
    for produce_patches_process in produce_patches_process_list:
        produce_patches_process.daemon = True
        produce_patches_process.start()
        
    return produce_patches_process_list, process_num, total_patch_num

def join_reader_processes():
    global produce_patches_process_list, finished_process_num
    is_done = False
    
    finished_process_num+=1   
    if finished_process_num == total_process_num:
        for produce_patches_process in produce_patches_process_list:
            produce_patches_process.join(timeout=3)
            
        is_done = True
        
    return is_done

def close_reader_processes():
    global produce_patches_process_list
    
    for produce_patches_process in produce_patches_process_list:
        produce_patches_process.terminate()


from torch.utils.data import  Dataset


class Batch_Dataset(Dataset):
    def __init__(self,batch_list):
        super().__init__()
        self.data = batch_list
        self.patch_20x_list = []
        self.patch_5x_list = []
        self.loc_list = []
        self.parsed_batch()

    def __len__(self):
        return len(self.loc_list)

    def __getitem__(self, idx):

        image_20x = self.patch_20x_list[idx]

        image_5x = self.patch_5x_list[idx]

        tile_name = self.loc_list[idx]
        
        return image_20x, image_5x, tile_name

    def parsed_batch(self):
        for level_patch_dict in self.data:
            self.loc_list.append(level_patch_dict[2])
            self.patch_20x_list.append(level_patch_dict[1][0])
            self.patch_5x_list.append(level_patch_dict[1][1])

        self.loc_list = np.array(self.loc_list)
        self.patch_20x_list = np.array(self.patch_20x_list)
        self.patch_5x_list = np.array(self.patch_5x_list)
