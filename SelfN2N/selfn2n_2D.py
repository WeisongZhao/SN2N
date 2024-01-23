# -*- coding: utf-8 -*-

import os
import random
import skimage
import torch
import datetime
import numpy as np
from glob import glob
from skimage.io import imread, imsave
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from SelfN2N.AUnet_2D import AUnet
from SelfN2N.loss import L0Loss, SSIM, MS_SSIM

class SelfN2N():
    def __init__(self, dataset_name, tests_name, reg = 1, reg_sparse = 0, 
                    constrained_type = 'L1', lr = 2e-4, epochs = 100, train_batch_size = 32, 
                        ifadaptive_lr = False, test_batch_size = 1, img_res = (128, 128)):
        """
        SelfN2N neural network
        ------
        dataset_name
            Name of the dataset, used for training.
        test_name
            Name of the dataset, used for testing.
        reg
            Weight of self-constrained loss
            {default: 1}
        reg_sparse
            Weight of sparse loss
            {default: 0}
        constrained_type
            Type of self-constrained loss
            {default: 'L1'}
        lr
            Learning rate
            {default: 2e-4}
        epochs
            Total number of training epochs
            {default: 100}
        train_batch_size
            Training batch size
            {default: 32}
        ifadaptive_lr
            Whether to use adaptive learning rate.
            {default: False}
        test_batch_size
            Test batch size
            {default: 1}
        img_res
            patch size
            {default: (128, 128)}
        """
        self.dataset_name = dataset_name
        self.tests_name = tests_name
        self.reg = reg
        self.reg_sparse = reg_sparse
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AUnet(n_channels = 1, n_classes = 1, bilinear=True).to(self.device)
        self.img_res = img_res
        
        self.train_batch_size = train_batch_size

        self.epochs = epochs
        self.test_batch_size = test_batch_size
        self.superior_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr, betas = (0.5, 0.999))
        self.constrained_type = constrained_type

        if self.constrained_type == 'L1':
            self.constrained = torch.nn.L1Loss(reduction='mean') 
            self.criterion = torch.nn.L1Loss(reduction='mean')
        elif self.constrained_type == 'L0':
            self.constrained = L0Loss()
            self.criterion = L0Loss()
        elif self.constrained_type == 'SmoothL1':
            self.constrained = torch.nn.SmoothL1Loss(reduction='mean') 
            self.criterion = torch.nn.SmoothL1Loss(reduction='mean')        
        elif self.constrained_type == 'None': 
            self.reg = 0
            self.criterion = torch.nn.L1Loss(reduction='mean')

        self.ifadaptive_lr = ifadaptive_lr
        if self.ifadaptive_lr:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', 
                                                            factor = 0.5,patience = 10, verbose = True)        
    def train(self):
        start_time = datetime.datetime.now() 
        history = []
        test_metrics = []
        lr_ms =[]
        traindata_path = ('../DL_dataset_2D/SN2N/%s/train' %(self.dataset_name))
        path = glob(traindata_path + '/*.tif')
        batch_num = int(np.floor(len(path) / self.train_batch_size))
        img_gt_path = (( '../DL_dataset_2D/SN2N/%s/test/GT' %(self.dataset_name)))
        save_path = (( '../images_2D/%s/images' %(self.tests_name)))

        for epoch in range(self.epochs):
            i = 0
            for inputs, labels in self.load_batch(traindata_path): # dataloader 是先shuffle后mini_batch  
                inputs = torch.from_numpy(inputs)
                labels = torch.from_numpy(labels)
                inputs, labels = inputs.to(self.device, dtype = torch.float32), labels.to(self.device, dtype = torch.float32)
                self.optimizer.zero_grad()                                
                inputs_pred1 = self.model(inputs)
                loss1 = self.criterion(inputs_pred1, labels)

                if self.reg != 0:
                    labels_pred1 = self.model(labels)                
                    loss2 = self.criterion(labels_pred1, inputs)
                    loss3 = self.constrained(labels_pred1, inputs_pred1)                   

                if self.reg_sparse == 0 and self.reg != 0:
                    loss = (loss1 + loss2 + self.reg * loss3)/(2 + self.reg)

                elif self.reg_sparse != 0 and self.reg != 0:
                    loss4 = self.criterion(torch.zeros_like(inputs_pred1), inputs_pred1) +\
                    self.criterion(labels_pred1, torch.zeros_like(labels_pred1))
                    loss = (loss1 + loss2 + self.reg * loss3 + self.reg_sparse * loss4) \
                    /(2 + self.reg + 2 * self.reg_sparse)

                elif self.reg_sparse != 0 and self.reg == 0:
                    loss4 = self.criterion(torch.zeros_like(inputs_pred1), inputs_pred1)
                    loss = (loss1 + self.reg_sparse * loss4) /(1 + self.reg_sparse)

                else:
                    loss = loss1

                total_loss = loss.item()  
                historym = np.array(total_loss)
                history.append(historym) 
                elapsed_time = datetime.datetime.now() - start_time
                print("[Epoch %d/%d] [Batch %d/%d] [loss:%f] time:%s" 
                      %(epoch, self.epochs, i, batch_num, total_loss * 100, elapsed_time))
                loss.backward()
                self.optimizer.step()            
                i = i + 1
                 
            if self.ifadaptive_lr:                                                 
                self.scheduler.step(total_loss)
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            lr_m = np.array(lr)
            lr_ms.append(lr_m) 


            test_path = ( '../DL_dataset_2D/SN2N/%s/test' %(self.dataset_name))

            if self.test_batch_size > 0:
                torch.cuda.empty_cache()
                test_pred = self.test(test_path)  
                test_pred = test_pred.to(torch.device("cpu"))
                test_pred = test_pred.numpy()
                self.saveResult(epoch, save_path, test_pred)

                if epoch % 10 == 0:
                    torch.save(self.model,  ('../images_2D/%s/weights/model_%d_%d_%d.pth')
                               %(self.tests_name, datetime.datetime.now().month, datetime.datetime.now().day, epoch)) 
            
        torch.save(self.model,  ('../images_2D/%s/weights/model_%d_%d_full.pth')
                       %(self.tests_name, datetime.datetime.now().month, datetime.datetime.now().day)) 
        f1 = open(('../images_2D/%s/weights/loss.txt') %(self.tests_name), 'w')
        for i in history:
            f1.write(str(i)+'\r\n')
        f1.close()
        
        f2 = open(('../images_2D/%s/weights/lr.txt') %(self.tests_name), 'w')
        for i in lr_ms:
            f2.write(str(i)+'\r\n')
        f2.close()
        
    def test(self, test_path):        
        with torch.no_grad():
            for data in self.load_test_batch(test_path):
                data = torch.from_numpy(data)
                data= data.to(self.device, dtype = torch.float32)
                y_pred = self.model(data)
                return y_pred  
            
    def saveResult(self, epoch, save_path, image_arr):
        for i, item in enumerate(image_arr):
            item = self.normalize(item)
            imsave(os.path.join(save_path, "epoch_%d.tif" %(epoch)), item)

    def normalize(self, stack):
        stack = stack.astype('float32')
        stack = stack - np.min(stack)
        stack = stack / np.max(stack)
        return stack  
    
    def load_batch(self, traindata_path):  
        path = glob(traindata_path + '/*.tif')
        batch_num = int(np.floor(len(path) / self.train_batch_size))
        imsize = (self.train_batch_size, 1, self.img_res[0], self.img_res[1])
        for i in range(batch_num):
            location = np.array(np.random.randint(low = 0, high=len(path), size=(self.train_batch_size, 1), dtype = 'int'))
            for batchsize in range(self.train_batch_size):
            # location = location.tolist()
                batch = []
                batch_tem = path[int(location[batchsize, :])]
                batch.append(batch_tem)
                imgs_As = []
                imgs_Bs = []
                for img in batch:
                    img = imread(img)
                    h, w = img.shape
                    half_w = int(w/2)
                    img_data = img[:, :half_w]
                    img_label = img[:, half_w:]
                    a = np.random.random()
                    b = np.random.random()
                    if a < 0.5:
                        img_data = np.fliplr(img_data)
                        img_label = np.fliplr(img_label)
                    if a > 0.5:
                        img_data = np.flipud(img_data)
                        img_label = np.flipud(img_label)
                    if b < 0.33:
                        img_data = np.rot90(img_data, 1)
                        img_label = np.rot90(img_label, 1)
                    if b > 0.33 and b < 0.66:
                        img_data = np.rot90(img_data, 2)
                        img_label = np.rot90(img_label, 2)
                    if b > 0.66:
                        img_data = np.rot90(img_data, 3)
                        img_label = np.rot90(img_label, 3)
                    img_label = img_label - np.min(img_label)
                    img_label = img_label / np.max(img_label)
                    img_data = img_data - np.min(img_data)
                    img_data = img_data / np.max(img_data)
                    img_data = img_data.astype('float32')
                    img_label = img_label.astype('float32')
                    img_data = img_data.reshape(1, h, half_w)
                    img_label = img_label.reshape(1, h, half_w)
                    imgs_As.append(img_data)
                    imgs_Bs.append(img_label)
            imgs_As = np.array(imgs_As)
            imgs_Bs = np.array(imgs_Bs)
            yield imgs_As, imgs_Bs
    
    def load_test_batch(self, test_path):
        path = glob(test_path + '/*.tif')
        batch_num = int(np.floor(len(path) / self.test_batch_size))
        img_tem = imread(path[0])
        h, w = img_tem.shape
        imsize = (self.test_batch_size, 1, h, w)
        imgs_A=np.zeros(imsize)
        for i in range(batch_num):
            batch = path[i*self.test_batch_size:(i+1)*self.test_batch_size]
            for img in batch:
                img = imread(img)
                h, w = img.shape
                img = img - np.min(img)
                img = img / np.max(img)
                img = img.astype('float32')
                img = img.reshape(1, h, w)
               
            imgs_A[i, :, :, :] = img
            yield imgs_A