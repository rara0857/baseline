import os
import torch
import numpy as np
import wandb
from matplotlib import pyplot as plt
from dataloader import TumorDataModule
from aug_dataloader import Aug_TumorDataModule
from model import net_20x_5x
from config import config
from datetime import datetime
from tqdm import tqdm
import torch.nn as nn
import time
import sys
from transform import random_choose_op, remedy_setting, check_weight, adjust_weight


class ClassifyModel():
    def __init__(self, config, max_iterator_num):
        super().__init__()

        self.config = config
        self.project = os.getcwd().split('/')[3]
        self.train_iter = max_iterator_num
        # setup automatic mixed precision
        self.use_amp = True
        # setup logging
        self.init_setting()

        # load dataset
        self.data_loader()

        # load model
        self.num_class = self.config['num_class']
        self.model = net_20x_5x(num_classes=self.num_class)
        
        if torch.cuda.device_count() > 1:
            print("Using Multiple GPU", torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)
            
        self.model.to(self.device)

        # define loss function
        self.ce = nn.CrossEntropyLoss()

        # define optimizers and scheduals
        self.set_optimizer()
        self.flag = True
        self.val_iter = 100
        self.unlabel_iter = 20
        self.pseudo_list = self.read_file_list('../unlabel_list.txt')
        self.pseudo_count = 0

    def data_loader(self):
        dataloader = TumorDataModule(self.config)
        aug_dataloader = Aug_TumorDataModule(self.config)
        if self.config['is_train'] == True:
            self.train_dataloader = dataloader.train_dataloader()
            self.val_dataloader = dataloader.val_dataloader()
            self.unlabel_dataloader = aug_dataloader.unlabel_dataloader()

    def read_file_list(self, filename):
        file_list = []
        try:
            with open(filename, 'r') as f:
                for n in f:
                    file_list.append(n.strip())
            return file_list
        except:
            print('[ERROR] Read file not found' + filename)
            return []

    def init_setting(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # logging
        MODEL_CKPT = self.config['log_string_dir']
        # create log/weight folder
        if not os.path.exists(MODEL_CKPT):
            os.system("mkdir " + MODEL_CKPT)
        if self.config['is_train'] == True:
            self.SAVE_MODEL = MODEL_CKPT

        self.best_weight_name = self.config["log_string_dir"] + self.config["checkpoint_path"]

        # set wandb
        self.wandb = wandb
        self.wandb.init(project = self.project)
        self.wandb.config.max_iterator_num = self.train_iter
        self.wandb.config.patch_size = self.config['patch_size']
        self.wandb.config.stride_size = self.config['stride_size']
        self.wandb.config.train_batch_size = self.config['train_batch_size']
        self.wandb.config.val_batch_size = self.config['val_batch_size']

    def set_optimizer(self):
        if self.config['re_train']:
            text = []
            with open(self.best_weight_name) as f:
                for line in f:
                    text.append(line)
            model_checkpoint_path = text[0]
            self.global_val_loss = float(text[1])

            self.step = int(model_checkpoint_path)
            best_weight = self.config['log_string_dir'] + f"model.ckpt-{self.step}.pt"
            checkpoint_weight = self.config["log_string_dir"]+best_weight
            checkpoint = torch.load(checkpoint_weight)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        else:
            self.step = 0
            self.global_val_loss = 0

        self.grad_scaler = torch.cuda.amp.GradScaler(enabled = self.use_amp)


        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.config['lr'])
#         decay_rate = 0.75
#         decay_steps = 7500
#         gamma = decay_rate ** (self.step / decay_steps)
#         self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=decay_steps, gamma=decay_rate) # update every 7500 steps
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0 = 2000, T_mult = 1)

    def forward_step(self, batch, istrain = True):
        img_high, img_low, label, case = batch
        if istrain:
            self.pseudo_count += sum(np.isin(np.array(case), np.array(self.pseudo_list)))
        img_high, img_low, label = img_high.to(self.device), img_low.to(self.device), label.to(self.device)
        label = np.squeeze(label)
        with torch.cuda.amp.autocast():
            prob = self.model(img_high, img_low)
            loss = self.ce(prob, label)
        return prob,label,loss
    
    def aug_forward_step(self, batch):
        image_20x_identity, image_5x_identity, image_20x_aug, image_5x_aug, label, case = batch
        image_20x_identity, image_5x_identity = image_20x_identity.to(self.device), image_5x_identity.to(self.device)
        image_20x_aug, image_5x_aug = image_20x_aug.to(self.device), image_5x_aug.to(self.device)

        prob_identity = self.model(image_20x_identity,image_5x_identity)
        prob_aug = self.model(image_20x_aug,image_5x_aug)

        return prob_identity, prob_aug, label

    def trainer(self):
        
        remedy_setting(3)
        train_dataloader_iter = iter(self.train_dataloader)
        valid_dataloader_iter = iter(self.val_dataloader)
        unlabel_dataloader_iter = iter(self.unlabel_dataloader)
        for iterator_num in range(0, self.train_iter + 1):
            self.model.train()
            try:
                batch = next(train_dataloader_iter)
            except StopIteration:
                train_dataloader_iter = iter(self.train_dataloader)
                batch = next(train_dataloader_iter)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # Training model by one batch.
            if self.flag:
                t_run_Start = time.time()
                self.flag = False
                
            prob ,label, loss = self.forward_step(batch)

            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            self.scheduler.step()

            acc = self.accuracy(prob, label)

            # write wandb
            self.wandb.log({"train/crossentropy": loss.item(), "step": iterator_num})
            self.wandb.log({"train/accuracy": acc, "step": iterator_num})
            self.wandb.log({"learning_rate":self.optimizer.param_groups[0]['lr'],"step":iterator_num})

            if iterator_num % 50 == 0:
                t_run_End = time.time()
                self.flag = True
                print('iterator num: ' + str(iterator_num + int(self.step / self.config['train_batch_size'])))
                print("\tRun cost %f sec" % (t_run_End - t_run_Start))
                l, P, N, TP, TN, FP, FN = self.get_metric(loss, prob, label)
                self.save_metric(l, P, N, TP, TN, FP, FN)
                print("pseudo label count : {} ".format(self.pseudo_count))
                print("label count : {} ".format((iterator_num + 1) * self.config['train_batch_size'] - self.pseudo_count))
                print('-----------------------------------------')
                   
            if iterator_num % 100 == 0:
                l_,P,T,N,TP,TN,FP,FN = 0,0,0,0,0,0,0,0
                strat = time.time()            
                for i in range(self.val_iter):
                    try:
                        batch = next(valid_dataloader_iter)
                    except StopIteration:
                        valid_dataloader_iter = iter(self.val_dataloader)
                        batch = next(valid_dataloader_iter)

                    # Testing batch data(Validation).
                    val_loss, val_prob, val_label = self.validation(batch)

                    batch_l, batch_P, batch_N, batch_TP, batch_TN, batch_FP, batch_FN = self.get_metric(val_loss, val_prob, val_label)

                    l_ += batch_l
                    P += batch_P
                    N += batch_N
                    TP += batch_TP
                    TN += batch_TN
                    FP += batch_FP
                    FN += batch_FN

                end = time.time()

                print('Validation:')
                print('\tcost time: ' + str(end - strat))

                # write to wandb
                self.wandb.log({"val/val_loss": l_ / self.val_iter, "step": iterator_num})
                self.wandb.log({"val/val_acc": (TP+TN)/(TP+TN+FP+FN), "step": iterator_num})

                self.save_metric( l_ / self.val_iter, P, N, TP, TN, FP, FN)
                if iterator_num == 0 and self.global_val_loss == 0:
                    train_batch_size = self.config['train_batch_size']
                    # save best weight
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                    }, self.SAVE_MODEL + "/" + "best_model.pt")
                    print("------SAVE INITIAL WEIGHT-------")
                    self.global_val_loss = l_

                    self.best_weight_name = self.config["log_string_dir"] + self.config["checkpoint_path"]
                    with open(self.best_weight_name, 'w') as f:
                        f.write(str((iterator_num + int(self.step / train_batch_size)) * train_batch_size))
                        f.write(str(self.global_val_loss))


                if l_ < self.global_val_loss:
                    train_batch_size = self.config['train_batch_size']
                    print('---------------SAVE MODEL-----------------')
                    # Save model.
                    print('\tSave model: ' + str((iterator_num + int(self.step / train_batch_size)) * train_batch_size))
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                    }, self.SAVE_MODEL + "/" + "model.ckpt-" + str((iterator_num + int(self.step / train_batch_size)) * train_batch_size)+ ".pt")
                    
                    self.global_val_loss = l_
                    print('-----------------------------------------')
                    # save best weight
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                    }, self.SAVE_MODEL + "/" + "best_model.pt")
                    
            if iterator_num % 200 == 0 and iterator_num > 0:
                
                count = 0
                diff = 0
                
                while(count < 10):
                    
                    ops = random_choose_op()
                    for i in range(self.unlabel_iter):
                        try:
                            unlabel_batch = next(unlabel_dataloader_iter)
                        except StopIteration:
                            unlabel_dataloader_iter = iter(self.unlabel_dataloader)
                            unlabel_batch = next(unlabel_dataloader_iter)

                        diff += self.aug_inference(unlabel_batch)
                    diff /= self.unlabel_iter
                    adjust_weight(ops, diff)
                    print('[diff] : ' + str(diff))
                    if (0.1 <= diff <= 0.4):
                        break
                    else:
                        count += 1
                        diff = 0
                    
                if (0.1 <= diff <= 0.4):
                    print("[final diff] : " + str(diff))
                else:
                    print("[count > 10]")
                    
        check_weight()


    def validation(self,batch):
        self.model.eval()
        with torch.no_grad():
            prob ,label, loss = self.forward_step(batch, istrain = False)
        return loss, prob, label

    def get_metric(self,loss, predictions, labels):
        predictions = predictions.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        pre = np.argmax(predictions, 1)
        one_hot_pre = np.zeros((pre.shape[0], 2))
        one_hot_pre[np.arange(pre.shape[0]), pre] = 1

        # --TP--
        label_pos_index = labels[:] == 1
        pre_pos_index = one_hot_pre[:, 1] == 1

        # --TN--
        label_neg_index = labels[:] == 0
        pre_neg_index = one_hot_pre[:, 0] == 1

        P = label_pos_index.sum()
        N = label_neg_index.sum()

        TP = (label_pos_index * pre_pos_index).sum()
        TN = (label_neg_index * pre_neg_index).sum()
        FP = N - TN
        FN = P - TP

        return float(loss), float(P), float(N), float(TP), float(TN), float(FP), float(FN)

    def save_metric(self,loss, P, N, TP, TN, FP, FN):
        accuracy = (TP + TN) / (P + N)

        if P == 0:
            sensitivity = 0.0
        else:
            sensitivity = TP / P

        if N == 0:
            false_positive_rate = 0
        else:
            false_positive_rate = FP / N

        if P + (TP + FP) == 0:
            F1_score = 0
        else:
            F1_score = (2 * TP) / (P + (TP + FP))

        print('\tBatch Loss = %.2f\t Accuracy = %.2f' % (loss, accuracy))
        print('\tSensitivity = %.2f\t FPR = %.2f' % (sensitivity, false_positive_rate))
        print('\tF1 score = %.2f' % (F1_score))
        return

    def accuracy(self,out, labels):
        _, pred = torch.max(out, dim=1)
        total = np.shape(labels)[0]
        return 100 * torch.sum(pred == labels).item() / total
    
    #PCE 
    def aug_inference(self,batch):
        self.model.eval()
        softmax_layer = nn.Softmax(dim = 1)
        with torch.no_grad():
            prob_identity, prob_aug, label = self.aug_forward_step(batch)
            
        prob_identity_softmax = softmax_layer(prob_identity).cpu().detach().numpy()
        prob_aug_softmax = softmax_layer(prob_aug).cpu().detach().numpy()
        
        tumor_confidence_identtity = prob_identity_softmax[:,1]
        tumor_confidence_aug = prob_aug_softmax[:,1]

        diff = sum(abs(np.array(tumor_confidence_identtity) - np.array(tumor_confidence_aug))) / self.config["unlabel_batch_size"]
        return diff
#         var =  sum(np.var([tumor_confidence_identtity, tumor_confidence_aug],axis = 0)) / self.config["unlabel_batch_size"]
#         print('var : ' + str(var))f
#         return var


if config['is_train'] == True:
#     max_iterator_num = 6000
    max_iterator_num = int(sys.argv[1])
    t_Start = time.time()
    ClassifyModel(config, max_iterator_num).trainer()
    t_End = time.time()
    print("Finish Training: ", t_End-t_Start)