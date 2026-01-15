# -*- coding: utf-8 -*-

import os
import torch
import datetime
import tifffile
import numpy as np
from glob import glob
from skimage.io import imread, imsave
from SN2N.models import Unet_2d, Unet_3d
from SN2N.utils import normalize

class net2D():
    def __init__(self, img_path,  sn2n_loss = 1, bs = 32, lr = 2e-4, epochs = 100, 
                 img_patch = '128', if_alr = True):
        """
        Self-inspired Noise2Noise
        -----Parameters------
        =====Important==========
        img_path:
            Path of raw images to train.
        sn2n_loss:
            Weight of self-constrained loss.
            {default: 1}
        bs:
            Training batch size.
            {default: 32}
        lr:
            Learning rate
            {default: 2e-4}.
        epochs:
            Total number of training epochs.
            {default: 100}.
            
        =====No need to change=====
        img_patch
            patch size
            {default: '128'}
        if_alr
            Whether to use adaptive learning rate.
            {default: True}
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_path = img_path
        self.parent_dir = os.path.dirname(img_path)
        self.dataset_path = os.path.join(self.parent_dir, 'datasets')
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        self.model_save_path = os.path.join(self.parent_dir, 'models')
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        self.images_path = os.path.join(self.parent_dir, 'images')
        if not os.path.exists(self.images_path):
            os.makedirs(self.images_path)
        self.sn2n_loss = sn2n_loss
        self.model = Unet_2d(n_channels = 1, n_classes = 1, bilinear=True).to(self.device)
        self.bs = bs
        self.epochs = epochs
        self.lr = lr
        self.img_patch = (int(img_patch),) * 2
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr, betas = (0.5, 0.999))
        self.constrained = torch.nn.L1Loss(reduction='mean') 
        self.criterion = torch.nn.L1Loss(reduction='mean')
        self.if_alr = if_alr
        if self.if_alr:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', 
                                                            factor = 0.5,patience = 10, verbose = True)  
         
    def train(self):
        print('The path for the raw images used for training is located under:\n%s' %(self.img_path))
        print('The training dataset is being saved under:\n%s' %(self.dataset_path))
        print('Models is being saved under:\n%s' %(self.model_save_path))
        print('Training temporary prediction images is being saved under:\n%s' %(self.images_path))
        start_time = datetime.datetime.now() 
        history = []
        test_metrics = []
        lr_ms =[]
        path = glob(self.dataset_path + '/*.tif')
        batch_num = int(np.floor(len(path) / self.bs))

        for epoch in range(self.epochs):
            i = 0
            for inputs, labels in self.load_batch2d(self.dataset_path):  
                inputs = torch.from_numpy(inputs)
                labels = torch.from_numpy(labels)
                inputs, labels = inputs.to(self.device, dtype = torch.float32), labels.to(self.device, dtype = torch.float32)
                self.optimizer.zero_grad()                                
                inputs_pred1 = self.model(inputs)
                loss1 = self.criterion(inputs_pred1, labels)

                if self.sn2n_loss != 0:
                    labels_pred1 = self.model(labels)                
                    loss2 = self.criterion(labels_pred1, inputs)
                    loss3 = self.constrained(labels_pred1, inputs_pred1)                   
                    loss = (loss1 + loss2 + self.sn2n_loss * loss3)/(2 + self.sn2n_loss)
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
                 
            if self.if_alr:                                                 
                self.scheduler.step(total_loss)
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            lr_m = np.array(lr)
            lr_ms.append(lr_m) 
            
            raw_path = glob(self.img_path + '/*.tif')
            test_img = tifffile.imread(raw_path[0])
            if len(test_img.shape) == 3:
                test_img = test_img[0, :, :]
            test_img = np.squeeze(test_img)
            torch.cuda.empty_cache()
            test_pred = self.test(test_img)  
            test_pred = test_pred.to(torch.device("cpu"))
            test_pred = test_pred.numpy()
            for i, item in enumerate(test_pred):
                item = normalize(item)
                tifffile.imsave(os.path.join(self.images_path, "epoch_%d.tif" %(epoch)), item)
            
            if epoch % 10 == 0:
                torch.save(self.model,  ('%s/model_%d_%d_%d.pth')
                           %(self.model_save_path, datetime.datetime.now().month, datetime.datetime.now().day, epoch)) 
            
        torch.save(self.model,  ('%s/model_%d_%d_full.pth')
                       %(self.model_save_path, datetime.datetime.now().month, datetime.datetime.now().day)) 
        
        f1 = open(('%s/loss.txt') %(self.parent_dir), 'w')
        for i in history:
            f1.write(str(i)+'\r\n')
        f1.close()
    
    def load_batch2d(self, traindata_path):  
        path = glob(traindata_path + '/*.tif')
        batch_num = int(np.floor(len(path) / self.bs))
        imsize = (self.bs, 1, self.img_patch[0], self.img_patch[1])
        for i in range(batch_num):
            location = np.array(np.random.randint(low = 0, high=len(path), size=(self.bs, 1), dtype = 'int'))
            for batchsize in range(self.bs):
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
            
    def test(self, test_path):        
       with torch.no_grad():
           for data in self.load_test_batch2d(test_path):
                data = torch.from_numpy(data)
                data= data.to(self.device, dtype = torch.float32)
                y_pred = self.model(data)
                return y_pred  
            
    def load_test_batch2d(self, img_tem):
        img_tem = np.squeeze(img_tem)
        h, w = img_tem.shape
        imsize = (1, 1, h, w)
        imgs_A = np.zeros(imsize)
        img_tem = normalize(img_tem)
        img_tem = img_tem.reshape(1, 1, h, w)
        imgs_A[:, :, :, :] = img_tem
        yield imgs_A
        
    
class net3D():
    def __init__(self, img_path,  sn2n_loss = 1, bs = 4, lr = 2e-4, epochs = 100, 
                 vol_patch = '16,128,128', if_alr = True):
        """
        Self-inspired Noise2Noise
        -----Parameters------
        =====Important==========
        img_path:
            Path of raw images to train.
        sn2n_loss:
            Weight of self-constrained loss.
            {default: 1}
        bs:
            Training batch size.
            {default: 4}
        lr:
            Learning rate
            {default: 2e-4}.
        epochs:
            Total number of training epochs.
            {default: 100}.
            
        =====No need to change=====
        vol_patch
            patch size
            {default: '16,128,128'}
        if_alr
            Whether to use adaptive learning rate.
            {default: True}
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_path = img_path
        self.parent_dir = os.path.dirname(img_path)
        self.dataset_path = os.path.join(self.parent_dir, 'datasets')
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        self.model_save_path = os.path.join(self.parent_dir, 'models')
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        self.images_path = os.path.join(self.parent_dir, 'images')
        if not os.path.exists(self.images_path):
            os.makedirs(self.images_path)
        self.sn2n_loss = sn2n_loss
        self.model = Unet_3d(n_channels = 1, n_classes = 1, bilinear=True).to(self.device)
        self.bs = bs
        self.epochs = epochs
        self.lr = lr
        self.vol_patch = tuple(map(int, vol_patch.split(',')))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr, betas = (0.5, 0.999))
        self.constrained = torch.nn.L1Loss(reduction='mean') 
        self.criterion = torch.nn.L1Loss(reduction='mean')
        self.if_alr = if_alr
        if self.if_alr:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', 
                                                            factor = 0.5,patience = 10, verbose = True)        
    def train(self):
        print('The path for the raw images used for training is located under:\n%s' %(self.img_path))
        print('The training dataset is being saved under:\n%s' %(self.dataset_path))
        print('Models is being saved under:\n%s' %(self.model_save_path))
        print('Training temporary prediction images is being saved under:\n%s' %(self.images_path))
        start_time = datetime.datetime.now() 
        history = []
        test_metrics = []
        lr_ms =[]
        path = glob(self.dataset_path + '/*.tif')
        batch_num = int(np.floor(len(path) / self.bs))

        for epoch in range(self.epochs):
            i = 0
            for inputs, labels in self.load_batch3d(self.dataset_path):  
                inputs = torch.from_numpy(inputs)
                labels = torch.from_numpy(labels)
                inputs, labels = inputs.to(self.device, dtype = torch.float32), labels.to(self.device, dtype = torch.float32)
                self.optimizer.zero_grad()                                
                inputs_pred1 = self.model(inputs)
                loss1 = self.criterion(inputs_pred1, labels)

                if self.sn2n_loss != 0:
                    labels_pred1 = self.model(labels)                
                    loss2 = self.criterion(labels_pred1, inputs)
                    loss3 = self.constrained(labels_pred1, inputs_pred1)                   
                    loss = (loss1 + loss2 + self.sn2n_loss * loss3)/(2 + self.sn2n_loss)
                else:
                    loss = loss1

                total_loss = loss.item()  
                historym = np.array(total_loss)
                history.append(historym) 
                elapsed_time = datetime.datetime.now() - start_time
                if i%100==0:
                    print("[Epoch %d/%d] [Batch %d/%d] [loss:%f] time:%s"
                      %(epoch, self.epochs, i, batch_num, total_loss * 100, elapsed_time))
                loss.backward()
                self.optimizer.step()            
                i = i + 1
                 
            if self.if_alr:                                                 
                self.scheduler.step(total_loss)
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            lr_m = np.array(lr)
            lr_ms.append(lr_m) 
            
            raw_path = glob(self.img_path + '/*.tif')
            test_img = tifffile.imread(raw_path[0])
            test_img_tem = test_img[0:16, 220:476, 300:556]
            
            torch.cuda.empty_cache()
            test_pred = self.test(test_img_tem)  
            test_pred = test_pred.to(torch.device("cpu"))
            test_pred = test_pred.numpy()
            for i, item in enumerate(test_pred):
                # item = normalize(item)
                item = item/255
                tifffile.imsave(os.path.join(self.images_path, "epoch_%d.tif" %(epoch)), item)
            
            if epoch % 10 == 0:
                torch.save(self.model,  ('%s/model_%d_%d_%d.pth')
                           %(self.model_save_path, datetime.datetime.now().month, datetime.datetime.now().day, epoch)) 
            
        torch.save(self.model,  ('%s/model_%d_%d_full.pth')
                       %(self.model_save_path, datetime.datetime.now().month, datetime.datetime.now().day)) 

    
    def load_batch3d(self, traindata_path):  #[bs,channel,t,h,w]
        path = glob(traindata_path + '/*.tif')
        batch_num = int(np.floor(len(path) / self.bs))
        imsize = (self.bs, 1, self.vol_patch[0], self.vol_patch[1], self.vol_patch[2])
        
        for i in range(batch_num):
            imgs_A=np.zeros(imsize)
            imgs_B=np.zeros(imsize)
            location = np.array(np.random.randint(low = 0, high=len(path), size=(self.bs, 1), dtype = 'int'))
            for batchsize in range(self.bs):
                batch = []
                batch_tem = path[int(location[batchsize, :])]
                batch.append(batch_tem)
                
                for img in batch:#[0,32)
                    img = imread(img)
                    t, h, w = img.shape
                    half_w = int(w/2)
                    img_data = img[:, :, :half_w]
                    img_label = img[:, :, half_w:]
                    a = np.random.random()
                    b = np.random.random()
                    img_data_list = []
                    img_label_list = []
                    img_data_list_2 = []
                    img_label_list_2 = []
                    if a < 0.5:
                        for taxial in range(t):
                            img_data_temp = np.fliplr(img_data[taxial, :, :])
                            img_label_temp = np.fliplr(img_label[taxial, :, :])
                            img_data_list.append(img_data_temp)
                            img_label_list.append(img_label_temp)
                    else:
                        for taxial in range(t):
                            img_data_temp = np.flipud(img_data[taxial, :, :])
                            img_label_temp = np.flipud(img_label[taxial, :, :])
                            img_data_list.append(img_data_temp)
                            img_label_list.append(img_label_temp)
                    
                    img_data_list = np.array(img_data_list)
                    img_label_list = np.array(img_label_list)
                    if b < 0.33:
                        for taxial in range(t):
                            img_data_temp = np.rot90(img_data_list[taxial, :, :], 1)
                            img_label_temp = np.rot90(img_label_list[taxial, :, :], 1)
                            img_data_list_2.append(img_data_temp)
                            img_label_list_2.append(img_label_temp)
                    elif b > 0.33 and b < 0.66:
                        for taxial in range(t):
                            img_data_temp = np.rot90(img_data_list[taxial, :, :], 2)
                            img_label_temp = np.rot90(img_label_list[taxial, :, :], 2)
                            img_data_list_2.append(img_data_temp)
                            img_label_list_2.append(img_label_temp)
                    else:
                        for taxial in range(t):
                            img_data_temp = np.rot90(img_data_list[taxial, :, :], 3)
                            img_label_temp = np.rot90(img_label_list[taxial, :, :], 3)
                            img_data_list_2.append(img_data_temp)
                            img_label_list_2.append(img_label_temp)
                    
                    img_data_list_2 = np.array(img_data_list_2)
                    img_label_list_2 = np.array(img_label_list_2)
                    img_data_list_2 = img_data_list_2/255
                    img_label_list_2 = img_label_list_2/255
                    #img_data_list_2 = normalize(img_data_list_2)
                    #img_label_list_2 = normalize(img_label_list_2)
                    imgs_A[batchsize, 0, :, :, :] = img_data_list_2
                    imgs_B[batchsize, 0, :, :, :] = img_label_list_2
            yield imgs_A, imgs_B
            
    def test(self, img_tem):        
        with torch.no_grad():
            for data in self.load_test_batch3d(img_tem):
                data = torch.from_numpy(data)
                data= data.to(self.device, dtype = torch.float32)
                y_pred = self.model(data.contiguous())
                return y_pred 
            
            
    def load_test_batch3d(self, img):
        t, h, w = img.shape
        imsize = (1, 1, t, h, w)
        imgs_A=np.zeros(imsize)
        img_list = []
        for taxial in range(t):
            img_tem = img[taxial, :, :]
            img_tem = np.squeeze(img_tem)
            img_tem = img_tem/255
            # img_tem = normalize(img_tem)
            img_list.append(img_tem)
        img_list = np.array(img_tem)
        imgs_A[:, :, :, :, :] = img
        yield imgs_A
    
