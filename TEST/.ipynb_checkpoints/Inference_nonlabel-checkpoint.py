import sys,os

# import data reader API
# Need to be checked when the platform was changed
# sys.path.append('/opt/ASAP/bin')
# data reader API
# import multiresolutionimageinterface as mir
import openslide
import cv2
from multiprocessing import Lock, Queue
import multiprocessing as mp

# from WriteAnnotationJson import mask_2_ASAP_Json
from ReadBatchFromPyramidMP import *
import numpy as np
import time
from datetime import datetime
import json
import torch
from config_test import config

# model code
from model import net_20x_5x
from torch.utils.data import DataLoader

# Algorithm Adapter
# from abstractalgorithmserver.abstractalgorithmserver import AbstractAlgorithmServer


# class TumorDtection(AbstractAlgorithmServer):
class Testing():
    def __init__(self):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.patch_size = 256
        self.tile_size = 128
        self.train_batch_size = 16
        self.valid_batch_size = 20
        self.end_flag = False

        self.process_num = 3
        self.image_path = ''
        self.tifname = ''
        self.output_path = ''
        self.model_path = ''

        self.SHOW_IMG_ARGMAX = None
        self.show_img_GT = None
        self.show_img_tumor = None

    '''
    Read data from pyramid tif code.
    '''

    def geometric_series_sum(self, a, r, n):
        return a * (1.0 - pow(r, n)) / (1.0 - r)

    def parse_tile_loc(self, loc_str):
        tile_loc = loc_str.split('-')
        return int(int(tile_loc[0]) / self.tile_size), int(int(tile_loc[1]) / self.tile_size)

    def show_testing_img(self, locs_str, predictions):

        for i in range(predictions.shape[0]):
            if sys.version_info[0] < 3:
                tile_loc = self.parse_tile_loc(locs_str[i].decode("utf-8"))
            else:
                tile_loc = self.parse_tile_loc(locs_str[i])
#             print(predictions[i].item())
            self.SHOW_IMG_ARGMAX[tile_loc[1], tile_loc[0]] = predictions[i].item()


    
    
    def get_parsed_batch(self, batch_list):
        loc_list = []
        patch_20x_list = []
        patch_5x_list = []
        for level_patch_dict in batch_list:
            loc_list.append(level_patch_dict[2])
            patch_20x_list.append(np.array(level_patch_dict[1][0]))
            patch_5x_list.append(np.array(level_patch_dict[1][1]))


        patch_20x = torch.Tensor(patch_20x_list)
        patch_5x = torch.Tensor(patch_5x_list)
        loc = loc_list
            
        return patch_20x, patch_5x, loc

    '''
    Tumor classifier code.
    '''

    def tumor_classifier(
            self,
            testing_all_img=True):


        num_classes = 2

        '''
            read multi-magnification patches
        '''
        print("Setting up DataReader...")
        pyramid_level_set = [0, 1]
        batch_size = self.valid_batch_size
        patch_size = self.patch_size
        stride_size = self.tile_size
        process_num = self.process_num
        finished_process_num = 0
        mr_data = init_reader(self.image_path, type='none')

        # setup model
        net = net_20x_5x(num_classes)

        if mr_data is not None:
            lock = Lock()
            patches_queue = Queue(32)

            produce_patches_process_list, process_num, self.total_patch_num = read_batch_patches(self.image_path,
                                                                                                 pyramid_level_set,
                                                                                                 patches_queue,
                                                                                                 lock, process_num,
                                                                                                 batch_size,
                                                                                                 patch_size,
                                                                                                 stride_size)

            print(mp.active_children())
            print("Setting up Saver...")
            print(self.model_path)
            loader = torch.load(self.model_path)
            net.load_state_dict(loader,strict=False)
            net.to(self.device)
            print("To device")
            net.eval()
            
            print("Model restored...")

            self.processed_batch_num = 0

            if testing_all_img:

                while True:
                    batch_list = patches_queue.get()

                    if not mp.active_children():
                        break

                    if batch_list == 'DONE':
                        print(batch_list)
                        print(mp.active_children())
                        finished_process_num += 1

                        time.sleep(2)
                        if join_reader_processes():
                            break
                        continue

                    image_20xs, image_5xs, locs_str = self.get_parsed_batch(batch_list)
                    

                    t_Start = time.time()
                    # image_20xs, image_5xs = image_20xs.to(self.device), image_5xs.to(self.device)
                    image_20xs, image_5xs = image_20xs.cuda(), image_5xs.cuda()
                    output_feature = net(image_20xs, image_5xs)
#                     print(output_feature)
                    t_End = time.time()

                    l = torch.argmax(output_feature, 1)

                    self.processed_batch_num += 1

                    self.show_testing_img(locs_str, l)

                    if self.processed_batch_num % 100 == 0:
                        print('Image num =' + str(batch_size * self.processed_batch_num))
                        print('\tCost =' + str(t_End - t_Start))
                        print('----------------------------------------------')


    def check_init(self):

        return_flag = True

#         if not os.path.isfile(self.image_path):
#             print('Image is not exit.')
#             return_flag = False
#         if not os.path.isdir(self.model_path[:-1]):
#             print('Model is not exit at ' + self.model_path)
#             return_flag = False
#         if self.process_num <= 0:
#             print('PROCESS_NUM must large than 1.')
#             return_flag = False

        return return_flag

    def init_parameters(self, image_path, preprocessed_result_path, tifname, model_path):

        self.image_path = image_path
        self.output_path = preprocessed_result_path
        self.tifname = tifname
        self.model_path = model_path




    def evaluation(self, length, width):
        if self.SHOW_IMG_ARGMAX.max() != 0:
            self.SHOW_IMG_ARGMAX = self.SHOW_IMG_ARGMAX * (255.0 / self.SHOW_IMG_ARGMAX.max())
        self.SHOW_IMG_ARGMAX = np.uint8(self.SHOW_IMG_ARGMAX)
        if self.show_img_GT.max() != 0:
            self.show_img_GT = self.show_img_GT * (255.0 / self.show_img_GT.max())
        self.show_img_GT = np.uint8(self.show_img_GT)
        show_img_argMax = self.SHOW_IMG_ARGMAX[: length, : width]
        show_img_GT = self.show_img_GT[: length, : width]
        show_img_tumor = self.show_img_tumor

        # label_predict
        tn = np.where((show_img_argMax == 0) & (show_img_GT == 0))
        fp = np.where((show_img_argMax == 255) & (show_img_GT == 0))

        tp = np.where((show_img_argMax == 255) & (show_img_GT == 255))
        fn = np.where((show_img_argMax == 0) & (show_img_GT == 255))

        normal_label = len(tn[0]) + len(fp[0])
        tumor_label = len(tp[0]) + len(fn[0])

        normal_predict = len(tn[0]) + len(fn[0])
        tumor_predict = len(tp[0]) + len(fp[0])

        n = normal_label + tumor_label
        p0 = float((len(tn[0]) + len(tp[0])) / n)
        pc = float((normal_label * normal_predict + tumor_label * tumor_predict) / (n * n))
        if p0 == pc:
            k = 0
        else:
            k = (p0 - pc) / (1 - pc)

        # ---------------image---------------------
        show_img_tumor[tp] = [255, 255, 255]
        show_img_tumor[fp] = [0, 255, 0]
        show_img_tumor[fn] = [255, 0, 0]

        show_img_tumor = np.uint8(show_img_tumor)
        import IPython.display as dpy
        import PIL.Image
        #         if display_flag:
        #             dpy.display(PIL.Image.fromarray(show_img_tumor))

        TP_score = len(tp[0])
        TN_score = len(tn[0])
        FP_score = len(fp[0])
        FN_score = len(fn[0])

        if TP_score == 0 or (TP_score + FP_score + FN_score) == 0:
            IOU_score = 0
        else:
            IOU_score = float(TP_score / (TP_score + FP_score + FN_score))

        if TP_score == 0 or (TP_score + FN_score) == 0:
            Sen_score = 0
        else:
            Sen_score = float(TP_score / (TP_score + FN_score))

        if TP_score == 0 or (2 * TP_score + FP_score + FN_score) == 0:
            F1_score = 0
        else:
            F1_score = float(2 * TP_score / (2 * TP_score + FP_score + FN_score))
        kappa_score = k
        delimiter_ = ','
        score_list = []
        score_list.append(self.tifname)
        score_list.append(TP_score)
        score_list.append(TN_score)
        score_list.append(FP_score)
        score_list.append(FN_score)
        score_list.append(IOU_score)
        score_list.append(Sen_score)
        score_list.append(F1_score)
        score_list.append(kappa_score)
        o_file = self.output_path + self.tifname + '_score.txt'
        f_o = open(o_file, "a+")
        for score in score_list:
            f_o.write(str(score) + delimiter_)
        now = datetime.now()
        dt_str = now.strftime("%d/%m/%Y %H:%M:%S")
        f_o.write(dt_str)
        f_o.write("\n")
        f_o.close()


        print('IOU Score: ' + str(IOU_score))
        print('Sensitivity: ' + str(Sen_score))

    def execute(self, name):
        try:
            if self.check_init():
                # Init model path.
                mr_gt = openslide.OpenSlide(self.image_path)
                width = int(mr_gt.dimensions[0] / 128) + 1
                length = int(mr_gt.dimensions[1] / 128) + 1

                img_shape = [length, width]

                self.SHOW_IMG_ARGMAX = np.zeros(shape=img_shape)
                self.show_img_tumor = np.zeros(shape=[length, width, 3])

                # Run classifier.
                self.tumor_classifier()

                # Close data reader processes.
                close_reader_processes()

                if self.SHOW_IMG_ARGMAX.max() != 0:
                    self.SHOW_IMG_ARGMAX *= 255.0 / self.SHOW_IMG_ARGMAX.max()
                self.SHOW_IMG_ARGMAX = np.uint8(self.SHOW_IMG_ARGMAX)

                mr_gt = openslide.OpenSlide(self.image_path)
                width = int(mr_gt.dimensions[0] / 128) + 1
                length = int(mr_gt.dimensions[1] / 128) + 1

                classify = self.SHOW_IMG_ARGMAX[: length, : width]
                result = np.array(classify)

                cv2.imwrite(preprocessed_result_path + name + '_result.png', result)

                print("==================Evaluation====================")
                # Evaluation
                self.evaluation(length, width)

                # Write xml.
                # mask_2_ASAP_Json(result, self.output_path + name + '_result.json')
        except:
            import traceback
            exception_type, exception_value, exception_traceback = sys.exc_info()
            error_log = traceback.format_exception(exception_type, exception_value, exception_traceback)
            raise Exception(str(error_log))


def check_path(path):
    if not os.path.exists(path):
        print("create dictionary")
        os.system("mkdir "+path)


model_path = config['log_string_dir'] + config['best_weights']

for tifname in config['test_list']:
    image_path = '/home/u5914116/ALOVAS Tumor Detection Handover Code/DATA/RAW_DATA/RAW_IMAGE/{}.tif'.format(tifname)
    preprocessed_result_path = '/home/u5914116/ALOVAS Tumor Detection Handover Code/DATA/PROCESSED_DATA/CASE_UUID/{}/'.format(tifname)
    check_path(preprocessed_result_path)
    AlgorithmName_runner = Testing()
    AlgorithmName_runner.init_parameters(image_path, preprocessed_result_path, tifname, model_path)
    start = time.time()
    AlgorithmName_runner.execute(tifname)
    end = time.time()
    print(tifname, " is Finished and total time: ", str(end - start))

