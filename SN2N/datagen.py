# -*- coding: utf-8 -*-

import os
import numpy as np
import tifffile
from skimage.io import imsave
import random
from SN2N.utils import normalize
np.seterr(divide='ignore',invalid='ignore')


class generator2D():
    def __init__(self, img_path, P2Pmode = 0, P2Pup = 0, BAmode = 0, SWsize = 64, SWmode = 1, 
                 SWfilter = 1, P2Ppatch ='64', img_patch = '128', ifx2 = True, inter_mode = 'Fourier'):
        """
        Self-inspired Noise2Noise
        
        -----Parameters------
        =====Important==========
        img_path:
            Path of raw images to train.
        P2Pmode(0 ~ 3):
            Augmentation mode for Patch2Patch.
            0: NONE; 
            1: Direct interchange in t;
            2: Interchange in single frame;
            3: Interchange in multiple frame but in different regions;
            {default: 0}
        P2Pup:
            Increase the dataset to its (1 + P2Pup) times size.
            {default: 0}
        BAmode(0 ~ 2):
            Basic augmentation mode.
            0: NONE; 
            1: double the dataset with random rotate&flip;
            2: eightfold the dataset with random rotate&flip;
            {default: 0} 
        SWsize:
            Interval pixel of sliding window for generating image pathes.
            {default: 64}
            
        =====No need to change=====
        SWmode(0 ~ 1):
            Threshold mode to exclude some black patches.
            0: Set the actual threshold directly as Wth
            1: The actual threshold is set to the sum of the image's average value and the `SWfilter`.
            {default: 1}
        SWfilter(0 ~ 255):
            Threshold for excluding patches with black areas. 
            The actual threshold value is determined according to the description provided for `SWmode`.
            {default: 1}
        P2Ppatch:
            ROI size of interchange in Patch2Patch.
            {default: '64'}
        img_patch:
            Patch size.
            {default: '128'}
        ifx2:
            If re-scale to original size.
            True OR False
            {default: True}
        inter_mode:
            Scaling method.
            'Fourier': Fourier re-scaling;
            'bilinear': spatial re-scaling;
            {default: 'Fourier'}
		------
        """
        
        self.img_path = img_path
        self.parent_dir = os.path.dirname(img_path)
        self.dataset_path = os.path.join(self.parent_dir, 'datasets')
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        self.P2Pmode = P2Pmode
        self.P2Pup = int(P2Pup)
        self.P2P_times = int(np.sqrt(int(P2Pup)))
        self.P2P_rolls = int(np.sqrt(int(P2Pup)))
        if self.P2Pup > 0:
            self.P2P_rolls = max(1, self.P2P_rolls)
            self.P2P_times = max(1, self.P2P_times)
        self.BAmode = BAmode    
        self.SWsize = int(SWsize)
        self.SWmode = SWmode
        self.SWfilter = SWfilter
        self.P2Ppatch = (int(P2Ppatch),) * 2
        self.img_patch = (int(img_patch),) * 2
        self.img_patch_size = [int(img_patch), int(img_patch)]
        self.ifx2 = ifx2
        self.inter_mode = inter_mode
        
    
    def execute(self, flage = 1):
        """
        SN2N tool: Generator with random interchange 
        ------
        flage
            started file index
            {default: 1}

        Returns
        -------
        NULL
        """
        times = int(self.P2P_times)
        roll = int(self.P2P_rolls)
        print('The path for the raw images used for training is located under:\n%s' %(self.img_path))
        print('The training dataset is being saved under:\n%s' %(self.dataset_path))
        if self.P2Pmode == 0:
            times = 0
            roll = 0
        datapath_list = []
        for (root, dirs, files) in os.walk(self.img_path):
            for j, Ufile in enumerate(files):
                path = os.path.join(root, Ufile)
                datapath_list.append(path)
                  
        l = len(datapath_list)

        for ll in range(l):
            image_data = tifffile.imread(datapath_list[ll])
            print('For number %d frame'%(ll + 1))

            try:
                [t, x, y] = image_data.shape
                for taxial in range(t):
                    image_arr = self.slidingWindow2d(image_data[taxial,:,:])
                    flage = self.savedata2d(image_arr, flage)

                    for circlelarge in range(roll):
                        if times >= 1:
                            image_data_pre = self.random_interchange(imga = image_data[taxial,:],
                                imgb = image_data[random.randint(0, t - 1),:,:])                        
                            for circle in range(times - 1):
                                image_data_pre = self.random_interchange(imga = image_data_pre,
                                    imgb = image_data[random.randint(0, t - 1),:,:])
                            image_arr = self.slidingWindow2d(image_data_pre)
                            flage = self.savedata2d(image_arr, flage)

            except ValueError:
                if self.P2Pmode == 3:
                    image_data_b = tifffile.imread(datapath_list[random.randint(0, l - 1)])
                else:
                    image_data_b = []
                image_arr = self.slidingWindow2d(image_data)
                flage = self.savedata2d(image_arr, flage)
                
                for circlelarge in range(roll):
                    if times >= 1:
                        image_data_pre = self.random_interchange(imga = image_data,
                            imgb = image_data_b) 
                        for circle in range(times - 1):
                            image_data_pre = self.random_interchange(imga = image_data_pre,
                                imgb = image_data_b)
                        image_arr = self.slidingWindow2d(image_data_pre)
                        flage = self.savedata2d(image_arr, flage)
        return   
    
    def block2d(self, image_stack):
        """
        Self-supervised data generator in xy
        ------
        image_stack
            image TO generate
        
        Returns
        -------
        left, right: noisy data pair
        """
        image_stack = image_stack.astype('float32')        
        if image_stack.ndim == 2:        
            image_stack = np.expand_dims(image_stack, 0)
        if image_stack.ndim == 1: 
            imsize = self.img_patch_size
            imsize = (int(imsize[0] / 2), int(imsize[1] / 2))
            left = np.zeros((1, imsize[0], imsize[1]))
            right = np.zeros((1, imsize[0], imsize[1]))
            return left, right
        [t, x, y] = image_stack.shape 
        image_stack
        upleft = []
        upright = []
        downright = []
        downleft = []
        for i in range(t):
            ul = image_stack[i, 0::2, 0::2]
            ur = image_stack[i, 0::2, 1::2]
            dr = image_stack[i, 1::2, 1::2]
            dl = image_stack[i, 1::2, 0::2]
            upleft.append(ul)
            upright.append(ur)
            downright.append(dr)
            downleft.append(dl)
        left = np.array(upleft) / 2 + np.array(downright) / 2
        right = np.array(upright) / 2 + np.array(downleft) / 2
        
        return left, right 
    
    def slidingWindow2d(self, image_data):
        """
        SN2N tool: patch
        ------
        image_data
            image TO generate
        -------
        image_arr: patches with size (self.img_patch)
        """
        SWsize = self.SWsize
        SWmode = self.SWmode
        SWfilter = self.SWfilter
        img_patch = self.img_patch
        img_max =  np.iinfo(image_data.dtype).max
        image_data = 255*image_data.astype(np.float32) / img_max
        #image_data = 255*normalize(image_data)
        if SWmode == 0:
            threshold_real = SWfilter
        if SWmode == 1:
            avg = np.mean(image_data)
            threshold_real = avg + SWfilter
        bsize = img_patch[0]
        image_arr = []
        (h, w) = image_data.shape
        xx = int(np.floor(h - (bsize - SWsize)) / SWsize)
        yy = int(np.floor(w - (bsize - SWsize)) / SWsize)
        for i in range(1, (xx + 1)):
            for j in range(1, (yy + 1)):
                left1 = (j - 1) * SWsize
                right1 = (j - 1) * SWsize + bsize
                down = (i - 1) * SWsize
                up = (i - 1) * SWsize + bsize                    
                img = image_data[down:up, left1:right1]  
                if np.sum(img) > bsize * bsize * (threshold_real):
                    image_arr.append(img)
        image_arr = np.array(image_arr)
        return image_arr

    def basic_augment(self, img_data, mode):    
        """
        SN2N tool: Random flip&rotate
        ------
        img_data:
            image TO augmentation
        mode:
            Basic augmentation mode.
            0: NONE; 
            1: double the dataset with random rotate&flip;
            2: eightfold the dataset with random rotate&flip;
            {default: 0} 
        Returns
        -------
        img_data: image after flip&rotate
        """
        if mode == 1: 
            img_data = np.flipud(np.rot90(img_data)) 
        elif mode == 2: 
            img_data = np.flipud(img_data) 
        elif mode == 3: 
            img_data = np.fliplr(img_data) 
        elif mode == 4: 
            img_data = np.fliplr(np.rot90(img_data))
        elif mode == 5: 
            img_data = np.rot90(img_data)
        elif mode == 6:
            img_data = np.rot90(img_data, k = 2)
        elif mode == 7: 
            img_data = np.rot90(img_data, k = 3)
        return img_data
    

    def savedata2d(self, image_stack, flage):
        """
        SN2N tool: TO save data
            DATA structure: img, label (h, 2 x h) uint8
        ------
        image_stack
            data TO save
        flage  
            data number
        -------
        """
        
        left, right = self.block2d(image_stack)            
        if self.ifx2:
            imsize = self.img_patch
            if self.inter_mode == 'bilinear':
                left = self.imgstack_resize(left, imsize)
                right = self.imgstack_resize(right, imsize)
            elif self.inter_mode == 'Fourier':
                left = self.fourier_inter(left)
                right = self.fourier_inter(right)
        else:
            imsize = self.img_patch
            imsize = (int(imsize[0] / 2), int(imsize[1] / 2))
        [t, x, y] = left.shape
        size1 = (imsize[0], imsize[1] * 2)
        imgpart = np.zeros(size1, dtype = 'float32')
        imgpart_aug = np.zeros(size1, dtype = 'float32')
        imgpart_list = []
        for tt in range(t):
            temp_l = left[tt, :, :]
            temp_l = temp_l/255
            #temp_l = normalize(temp_l)
            temp_r = right[tt, :, :]
            temp_r = temp_r/255
            #temp_r = normalize(temp_r)
            imgpart[0 : imsize[0], 0 : imsize[1]] = temp_l        
            imgpart[0 : imsize[0], imsize[1] : 2 * imsize[1]] = temp_r
            imgpart_copy = imgpart.copy()
            imgpart_list.append(imgpart_copy)
            if self.BAmode == 1:
                mode = random.randint(0, 7)                
                temp_l_aug = self.basic_augment(temp_l, mode)
                temp_r_aug = self.basic_augment(temp_r, mode)
                imgpart_aug[0 : imsize[0], 0 : imsize[1]] = temp_l_aug        
                imgpart_aug[0 : imsize[0], imsize[1] : 2 * imsize[1]] = temp_r_aug
                imgpart_aug_copy = imgpart_aug.copy()
                imgpart_list.append(imgpart_aug_copy)
            elif self.BAmode == 2:
                for m in range(1, 8):
                    temp_l_aug = self.basic_augment(temp_l, m)
                    temp_r_aug = self.basic_augment(temp_r, m)
                    imgpart_aug[0 : imsize[0], 0 : imsize[1]] = temp_l_aug       
                    imgpart_aug[0 : imsize[0], imsize[1] : 2 * imsize[1]] = temp_r_aug
                    imgpart_aug_copy = imgpart_aug.copy()
                    imgpart_list.append(imgpart_aug_copy)          

        imgpart_list_copy = np.array(imgpart_list)
          
        [slices, x, y] = imgpart_list_copy.shape
        for s in range(slices):    
            img = imgpart_list_copy[s, :, :]
            img = img * 255
            if np.mean(img) > 0:                
                imsave(('%s/%d.tif') %(self.dataset_path, flage), img.astype('uint8'))
                flage = flage + 1
                if flage % 100 == 0:
                    print('Saving training images:', flage)
        return flage

    
    def fourier_inter(self, image_stack):
        """
        SN2N tool: Fourier re-scale
        ------
        image_stack
            image TO Fourier interpolation
        
        Returns
        -------
        imgf1: image with 2x size 
        """
        imsize = self.img_patch
        if image_stack.ndim == 2:        
            image_stack = np.expand_dims(image_stack, 0)
        [t, x, y] = image_stack.shape
        imgf1 = np.zeros((t, imsize[0], imsize[1]))
        
        for slice in range(t):
            img = image_stack[slice, :, :]
            imgsz = np.array([x, y])
            tem1 = np.divide(imgsz, 2)
            tem2 = np.multiply(tem1, 2)
            tem3 = np.subtract(imgsz, tem2)
            b = (tem3 == np.array([0, 0]))
            if b[0] == True:
                sz = imgsz - 1
            else:
                sz = imgsz            
            n = np.array([2, 2])
            ttem1 = np.add(np.ceil(np.divide(sz, 2)), 1)
            ttem2 = np.multiply(np.floor(np.divide(sz, 2)), np.subtract(n, 1))
            idx = np.add(ttem1, ttem2)
            padsize = np.array([x/2, y/2], dtype = 'int')
            pad_wid = np.ceil(padsize[0]).astype('int')
            img = np.pad(img, ((pad_wid, 0), (pad_wid, 0)), 'symmetric')
            img = np.pad(img, ((0, pad_wid), (0, pad_wid)),  'symmetric')
            imgsz1 = np.array(img.shape)
            tttem1 = np.multiply(n, imgsz1)
            tttem2 = np.subtract(n, 1)
            newsz = np.round(np.subtract(tttem1, tttem2))
            img1 = self.interpft(img, newsz[0], 0)
            img1 = self.interpft(img1, newsz[1], 1)
            idx = idx.astype('int')
            ttttem1 = np.subtract(np.multiply(n[0], imgsz[0]), 1).astype('int')
            ttttem2 = np.subtract(np.multiply(n[1], imgsz[1]), 1).astype('int')
            imgf1[slice, :, :] = img1[idx[0] - 1:idx[0] + ttttem1, idx[1] - 1:idx[1] + ttttem2]
            imgf1[imgf1 < 0] = 0
        return imgf1
    
    def interpft(self, x, ny, dim = 0):
        '''
        Function to interpolate using FT method, based on matlab interpft()
        ------
        x 
            array for interpolation
        ny 
            length of returned vector post-interpolation
        dim
            performs interpolation along dimension DIM
            {default: 0}

        Returns
        -------
        y: interpolated data
        '''
    
        if dim >= 1: 
        #if interpolating along columns, dim = 1
            x = np.swapaxes(x,0,dim)
        #temporarily swap axes so calculations are universal regardless of dim
        if len(x.shape) == 1:            
        #interpolation should always happen along same axis ultimately
            x = np.expand_dims(x,axis=1)
    
        siz = x.shape
        [m, n] = x.shape
    
        a = np.fft.fft(x,m,0)
        nyqst = int(np.ceil((m+1)/2))
        b = np.concatenate((a[0:nyqst,:], np.zeros(shape=(ny-m,n)), a[nyqst:m, :]),0)
    
        if np.remainder(m,2)==0:
            b[nyqst,:] = b[nyqst,:]/2
            b[nyqst+ny-m,:] = b[nyqst,:]
    
        y = np.fft.irfft(b,b.shape[0],0)
        y = y * ny / m
        y = np.reshape(y, [y.shape[0],siz[1]])
        y = np.squeeze(y)
    
        if dim >= 1:  
        #switches dimensions back here to get desired form
            y = np.swapaxes(y,0,dim)
    
        return y
    

    def random_interchange(self, imga, imgb = []):
        """
        SN2N tool: Random interchange
        ------
        imga
            image TO ROI interchange
        imgb
            another image for ROI interchange
            {default: []}
            
        Returns
        -------
        img: image after ROI interchange
        """
        mode = self.P2Pmode
        size = self.P2Ppatch
        if mode == 0:
            return imga

        if len(imgb) == 0:
            mode = 2    
        if mode == 1: #interchange along t-axial 
            img = self.interchange_multiple(imga, imgb, ifdirect = True)
        elif mode == 2: #interchange in single image
            img = self.interchange_single(imga)
        elif mode == 3: #interchange in different images
            img = self.interchange_multiple(imga, imgb, ifdirect = False)

        return img

    def interchange_multiple(self, imga, imgb, ifdirect = False):
        """
        SN2N tool: Core of random interchange in multiple images
        ------
        imga
            image TO ROI interchange
        imgb
            another image for ROI interchange
            {default: []}
        ifdirect
            interchange in same (True) or different (False) ROIs
            {default: False}

        Returns
        -------
        imga: image after ROI interchange
        """
        size = self.P2Ppatch
        h = size[0]
        w = size[1]
        if h < np.min((np.size(imga, 0), np.size(imgb, 0))) and w < np.min((np.size(imga, 1), np.size(imgb, 1))):
            xa = random.randint(0, np.size(imga, 0) - h)
            ya = random.randint(0, np.size(imga, 1) - w)
            if ifdirect:
                imga[xa: xa + h, ya : ya + w] = imgb[xa: xa + h, ya : ya + w] 
            else:
                xb = random.randint(0, np.size(imgb, 0) - h)
                yb = random.randint(0, np.size(imgb, 1) - w)
                imga[xa: xa + h, ya : ya + w] = imgb[xb : xb + h, yb : yb + w]

        return imga
    
    

    def interchange_single(self, img):
        """
        SN2N tool: Core of random interchange in single image
        ------
        img
            image TO ROI interchange

        Returns
        -------
        img: image after ROI interchange
        """
        size = self.P2Ppatch
        h = size[0]
        w = size[1]

        if h < np.size(img, 0) and w < np.size(img, 1):
            x1 = random.randint(0, np.size(img, 0) - h)
            y1 = random.randint(0, np.size(img, 1) - w)
            x2 = random.randint(0, np.size(img, 0) - h)
            y2 = random.randint(0, np.size(img, 1) - w)
            img[x1: x1 + h, y1 : y1 + w], img[x2 : x2 + h, y2 : y2 + w] = \
            img[x2 : x2 + h, y2 : y2 + w], img[x1: x1 + h, y1 : y1 + w]
        return img

    
    

    


class generator3D():
    def __init__(self, img_path, P2Pmode = 0, P2Pup = 0, BAmode = 0, SWsize = 64, SWmode = 1, 
                 SWfilter = 1, P2Ppatch ='64', vol_patch = '16,128,128', ifx2 = True, inter_mode = 'Fourier'):
        """
        Self-inspired Noise2Noise
        
        -----Parameters------
        =====Important==========
        img_path:
            Path of raw images to train.
        P2Pmode(0 ~ 3):
            Augmentation mode for Patch2Patch.
            0: NONE; 
            1: Direct interchange in t;
            2: Interchange in single frame;
            3: Interchange in multiple frame but in different regions;
            {default: 0}
        P2Pup:
            Increase the dataset to its (1 + P2Pup) times size.
            {default: 0}
        BAmode(0 ~ 2):
            Basic augmentation mode.
            0: NONE; 
            1: double the dataset with random rotate&flip;
            2: eightfold the dataset with random rotate&flip;
            {default: 0} 
        SWsize:
            Interval pixel of sliding window for generating image pathes.
            {default: 64}
            
        =====No need to change=====
        SWmode(0 ~ 1):
            Threshold mode to exclude some black patches.
            0: Set the actual threshold directly as Wth
            1: The actual threshold is set to the sum of the image's average value and the `SWfilter`.
            {default: 1}
        SWfilter(0 ~ 255):
            Threshold for excluding patches with black areas. 
            The actual threshold value is determined according to the description provided for `SWmode`.
            {default: 1}
        P2Ppatch:
            ROI size of interchange in Patch2Patch.
            {default: '64'}
        vol_patch:
            Patch size.
            {default: '16,128,128'}
        ifx2:
            If re-scale to original size.
            True OR False
            {default: True}
        inter_mode:
            Scaling method.
            'Fourier': Fourier re-scaling;
            'bilinear': spatial re-scaling;
            {default: 'Fourier'}
		------
        """
        
        self.img_path = img_path
        self.parent_dir = os.path.dirname(img_path)
        self.dataset_path = os.path.join(self.parent_dir, 'datasets')
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        self.P2Pmode = P2Pmode
        self.P2Pup = int(P2Pup)
        self.P2P_times = int(np.sqrt(int(P2Pup)))
        self.P2P_rolls = int(np.sqrt(int(P2Pup)))
        if self.P2Pup > 0:
            self.P2P_rolls = max(1, self.P2P_rolls)
            self.P2P_times = max(1, self.P2P_times)
        self.BAmode = BAmode 
        self.SWsize = SWsize
        self.SWmode = SWmode
        self.SWfilter = SWfilter
        self.P2Ppatch = (int(P2Ppatch),) * 2
        self.vol_patch = tuple(map(int, vol_patch.split(',')))
        self.ifx2 = ifx2
        self.inter_mode = inter_mode
        self.multi_frame = self.vol_patch[0]
        
    def execute(self, flage = 1):
        """
        SN2N tool: Generator with random interchange 
        ------
        flage
            started file index
        Returns
        -------
        NULL
        """
        print('The path for the raw images used for training is located under:\n%s' %(self.img_path))
        print('The training dataset is being saved under:\n%s' %(self.dataset_path))
        times = int(self.P2P_times)
        roll = int(self.P2P_rolls)
        multi_frame = self.vol_patch[0]
        if self.P2Pmode == 0:
            times = 0
            roll = 0
        datapath_list = []
        for (root, dirs, files) in os.walk(self.img_path):
            for j, Ufile in enumerate(files):
                path = os.path.join(root, Ufile)
                datapath_list.append(path)
        l= len(datapath_list)
        for ll in range(l):
            print('For number %d frame'%(ll + 1))
            image_data_stack = tifffile.imread(datapath_list[ll])
            
            image_data_stack_list = []
            [t, x, y] = image_data_stack.shape
            for tt in range(t-self.multi_frame):
                image_data = image_data_stack[tt : tt + self.multi_frame, :, :]
                image_data_stack_list.append(image_data)
            image_data_stack_list = np.array(image_data_stack_list)
            
            for ttt in range(t-self.multi_frame):
                img_temp = np.squeeze(image_data_stack_list[ttt, :, :, :])
                image_arr = self.slidingWindow3d(img_temp)
                flage = self.savedata3d(image_arr, flage)
            
            
            if self.P2Pmode == 1 or self.P2Pmode == 3:
                image_data_b = tifffile.imread(datapath_list[random.randint(0, l - 1)])
            elif self.P2Pmode ==2:
                image_data_b = []
                
            for circlelarge in range(roll):
                if times >= 1:
                    image_data_pre = self.random_interchange(imga = image_data_stack,
                        imgb = image_data_b) 
                    for circle in range(times - 1):
                        image_data_pre = self.random_interchange(imga = image_data_pre,
                            imgb = image_data_b)
                    image_data_stack_list_pre = []
                    for tt in range(t-self.multi_frame):
                        image_data_pre = image_data_pre[tt : tt + self.multi_frame, :, :]
                        image_data_stack_list_pre.append(image_data_pre)
                    image_data_stack_list_pre = np.array(image_data_stack_list)
                    
                    for ttt in range(t-self.multi_frame):
                        img_temp = np.squeeze(image_data_stack_list_pre[ttt, :, :, :])
                        image_arr = self.slidingWindow3d(img_temp)
                        flage = self.savedata3d(image_arr, flage)
            
        return 
    
    def block3d(self, image_stack):
        """
        Self-supervised data generator in xy
        ------
        image_stack
            image TO generate
        
        Returns
        -------
        left, right: noisy data pair
        """
        image_stack = image_stack.astype('float32')        
        if image_stack.ndim == 2:        
            image_stack = np.expand_dims(image_stack, 0)
        if image_stack.ndim == 1: 
            h = (self.vol_patch[1])
            w = (self.vol_patch[2])
            imsize = (int(h / 2), int(w / 2))
            left = np.zeros((1, imsize[0], imsize[1]))
            right = np.zeros((1, imsize[0], imsize[1]))
            return left, right
        [frame, t, x, y] = image_stack.shape 
        image_stack
        upleft = []
        upright = []
        downright = []
        downleft = []
        for i in range(frame):
            ul = image_stack[i, :, 0::2, 0::2]
            ur = image_stack[i, :, 0::2, 1::2]
            dr = image_stack[i, :, 1::2, 1::2]
            dl = image_stack[i, :, 1::2, 0::2]
            upleft.append(ul)
            upright.append(ur)
            downright.append(dr)
            downleft.append(dl)
        left = np.array(upleft) / 2 + np.array(downright) / 2
        right = np.array(upright) / 2 + np.array(downleft) / 2
        
        return left, right 
    
    def slidingWindow3d(self, image_data_stack):
        """
        SN2N tool: patch
        ------
        image_data_stack
            image TO generate
        Returns
        -------
        image_arr: patches with size (self.vol_patch)
        """
        SWsize = self.SWsize
        SWmode = self.SWmode
        SWfilter = self.SWfilter
        (t, h, w) = image_data_stack.shape
        img_max =  np.iinfo(image_data_stack.dtype).max
        image_data_stack = 255*image_data_stack.astype(np.float32) / img_max
        #image_data_stack = 255*normalize(image_data_stack)
        if SWmode == 0:
            threshold_real = SWfilter
        if SWmode == 1:
            avg_list = []
            for taxial in range(t):
                img = image_data_stack[taxial, :, :]
                avg = np.mean(img)
                avg_list.append(avg)
            avg_list = np.array(avg_list)
            avg = np.mean(avg_list)
            threshold_real = avg+SWfilter
        bsize = self.vol_patch[1]
        
        
        xx = int(np.floor(h - (bsize - SWsize)) / SWsize)
        yy = int(np.floor(w - (bsize - SWsize)) / SWsize)
        
        image_arr = []
        for i in range(1, (xx + 1)):
            for j in range(1, (yy + 1)):
                left1 = (j - 1) * SWsize
                right1 = (j - 1) * SWsize + bsize
                down = (i - 1) * SWsize
                up = (i - 1) * SWsize + bsize 
                img = image_data_stack[:, down:up, left1:right1]  
                if np.sum(img[0]) > bsize * bsize * (threshold_real):
                    image_arr.append(img)
        image_arr = np.array(image_arr)
        return image_arr
    
    def basic_augment(self, img_data, mode):    
        """
        SN2N tool: Random flip&rotate
        ------
        img_data
            image TO augmentation
        mode
            basic augmentation mode
            
        Returns
        -------
        img_data: image after flip&rotate
        """
        if mode == 1: 
            img_data = np.flipud(np.rot90(img_data)) 
        elif mode == 2: 
            img_data = np.flipud(img_data) 
        elif mode == 3: 
            img_data = np.fliplr(img_data) 
        elif mode == 4: 
            img_data = np.fliplr(np.rot90(img_data))
        elif mode == 5: 
            img_data = np.rot90(img_data)
        elif mode == 6:
            img_data = np.rot90(img_data, k = 2)
        elif mode == 7: 
            img_data = np.rot90(img_data, k = 3)
        return img_data
    
    def savedata3d(self, image_stack, flage):
        """
        SN2N tool: TO save data
            DATA structure: img, label (h, 2 x h) uint8
        ------
        image_stack
            data TO save
        flage  
            data number

        -------
        """
        left, right = self.block3d(image_stack) 
        if left.ndim==3:
            return flage
            
        [frame, t, x, y] = left.shape
        if self.ifx2:
            x = self.vol_patch[1]
            y = self.vol_patch[1]
            imsize = (x, y)
            if self.inter_mode == 'bilinear':
                left = self.imgstack_resize(left, imsize)
                right = self.imgstack_resize(right, imsize)
            elif self.inter_mode == 'Fourier':
                
                left_stack = []
                right_stack = []
                for f in range(frame):
                    left_stack_t = []
                    right_stack_t = []
                    for taxial in range(t):
                        left_temp = left[f, taxial, :, :]
                        right_temp = right[f, taxial, :, :]
                        left_temp = np.squeeze(left_temp)
                        right_temp = np.squeeze(right_temp)
                        left_temp = self.fourier_inter(left_temp)
                        right_temp = self.fourier_inter(right_temp)
                        left_temp = np.squeeze(left_temp)
                        right_temp = np.squeeze(right_temp)
                        left_stack_t.append(left_temp)
                        right_stack_t.append(right_temp)
                    left_stack_t = np.array(left_stack_t)
                    right_stack_t = np.array(right_stack_t)
                    left_stack.append(left_stack_t)
                    right_stack.append(right_stack_t)
                left_stack = np.array(left_stack)
                right_stack = np.array(right_stack)
        else:
            imsize = self.img_patch
            imsize = (int(imsize[0] / 2), int(imsize[1] / 2))
        [frame, t, x, y] = left_stack.shape
        size1 = (t, imsize[0], imsize[1] * 2)
        imgpart = np.zeros(size1, dtype = 'float32')
        imgpart_aug = np.zeros(size1, dtype = 'float32')
        imgpart_list = []
        for f in range(frame):
            temp_l = left_stack[f, :, :, :]
            temp_l = temp_l / 255
            #temp_l = normalize(temp_l)
            temp_r = right_stack[f, :, :, :]
            temp_r = temp_r / 255
            #temp_r = normalize(temp_r)
            imgpart[:, 0 : imsize[0], 0 : imsize[1]] = temp_l        
            imgpart[:, 0 : imsize[0], imsize[1] : 2 * imsize[1]] = temp_r
            imgpart_copy = imgpart.copy()
            imgpart_list.append(imgpart_copy)
            if self.BAmode == 1:
                mode = random.randint(0, 7)   
                for taxial in range(t-self.multi_frame):
                    temp_l_aug = self.basic_augment(temp_l[taxial, :, :], mode)
                    temp_r_aug = self.basic_augment(temp_r[taxial, :, :], mode)
                    imgpart_aug[taxial, 0 : imsize[0], 0 : imsize[1]] = temp_l_aug        
                    imgpart_aug[taxial, 0 : imsize[0], imsize[1] : 2 * imsize[1]] = temp_r_aug
                    imgpart_aug_copy = imgpart_aug.copy()
                    imgpart_list.append(imgpart_aug_copy)
            elif self.BAmode == 2:
                for m in range(1, 8):
                    for taxial in range(t-self.multi_frame):
                        temp_l_aug = self.basic_augment(temp_l[taxial, :, :], m)
                        temp_r_aug = self.basic_augment(temp_r[taxial, :, :], m)
                        imgpart_aug[taxial, 0 : imsize[0], 0 : imsize[1]] = temp_l_aug       
                        imgpart_aug[taxial, 0 : imsize[0], imsize[1] : 2 * imsize[1]] = temp_r_aug
                        imgpart_aug_copy = imgpart_aug.copy()
                        imgpart_list.append(imgpart_aug_copy)          

        imgpart_list_copy = np.array(imgpart_list)
          
        [frame, t, x, y] = imgpart_list_copy.shape
        for f in range(frame):    
            img = imgpart_list_copy[f, :, :, :]
            img = img * 255
            if np.mean(img) > 0:                
                imsave(('%s/%d.tif') %(self.dataset_path, flage), img.astype('uint8'))
                flage = flage + 1
                if flage % 100 == 0:
                    print('Saving training images:', flage)
        return flage
    
    def fourier_inter(self, image_stack):
        """
        SN2N tool: Fourier re-scale
        ------
        image_stack
            image TO Fourier interpolation
        
        Returns
        -------
        imgf1: image with 2x size 
        """
        x = self.vol_patch[1]
        y = self.vol_patch[1]
        imsize = (x, y)
        if image_stack.ndim == 2:        
            image_stack = np.expand_dims(image_stack, 0)
        [t, x, y] = image_stack.shape
        imgf1 = np.zeros((t, imsize[0], imsize[1]))
        
        for slice in range(t):
            img = image_stack[slice, :, :]
            imgsz = np.array([x, y])
            tem1 = np.divide(imgsz, 2)
            tem2 = np.multiply(tem1, 2)
            tem3 = np.subtract(imgsz, tem2)
            b = (tem3 == np.array([0, 0]))
            if b[0] == True:
                sz = imgsz - 1
            else:
                sz = imgsz            
            n = np.array([2, 2])
            ttem1 = np.add(np.ceil(np.divide(sz, 2)), 1)
            ttem2 = np.multiply(np.floor(np.divide(sz, 2)), np.subtract(n, 1))
            idx = np.add(ttem1, ttem2)
            padsize = np.array([x/2, y/2], dtype = 'int')
            pad_wid = np.ceil(padsize[0]).astype('int')
            img = np.pad(img, ((pad_wid, 0), (pad_wid, 0)), 'symmetric')
            img = np.pad(img, ((0, pad_wid), (0, pad_wid)),  'symmetric')
            imgsz1 = np.array(img.shape)
            tttem1 = np.multiply(n, imgsz1)
            tttem2 = np.subtract(n, 1)
            newsz = np.round(np.subtract(tttem1, tttem2))
            img1 = self.interpft(img, newsz[0], 0)
            img1 = self.interpft(img1, newsz[1], 1)
            idx = idx.astype('int')
            ttttem1 = np.subtract(np.multiply(n[0], imgsz[0]), 1).astype('int')
            ttttem2 = np.subtract(np.multiply(n[1], imgsz[1]), 1).astype('int')
            imgf1[slice, :, :] = img1[idx[0] - 1:idx[0] + ttttem1, idx[1] - 1:idx[1] + ttttem2]
            imgf1[imgf1 < 0] = 0
        return imgf1
    
    def interpft(self, x, ny, dim = 0):
        '''
        Function to interpolate using FT method, based on matlab interpft()
        ------
        x 
            array for interpolation
        ny 
            length of returned vector post-interpolation
        dim
            performs interpolation along dimension DIM
            {default: 0}

        Returns
        -------
        y: interpolated data
        '''
    
        if dim >= 1: 
        #if interpolating along columns, dim = 1
            x = np.swapaxes(x,0,dim)
        #temporarily swap axes so calculations are universal regardless of dim
        if len(x.shape) == 1:            
        #interpolation should always happen along same axis ultimately
            x = np.expand_dims(x,axis=1)
    
        siz = x.shape
        [m, n] = x.shape
    
        a = np.fft.fft(x,m,0)
        nyqst = int(np.ceil((m+1)/2))
        b = np.concatenate((a[0:nyqst,:], np.zeros(shape=(ny-m,n)), a[nyqst:m, :]),0)
    
        if np.remainder(m,2)==0:
            b[nyqst,:] = b[nyqst,:]/2
            b[nyqst+ny-m,:] = b[nyqst,:]
    
        y = np.fft.irfft(b,b.shape[0],0)
        y = y * ny / m
        y = np.reshape(y, [y.shape[0],siz[1]])
        y = np.squeeze(y)
    
        if dim >= 1:  
        #switches dimensions back here to get desired form
            y = np.swapaxes(y,0,dim)
    
        return y

    def random_interchange(self, imga, imgb = []):
        """
        SN2N tool: Random interchange
        ------
        imga
            image TO ROI interchange
        imgb
            another image for ROI interchange
            {default: []}
            
        Returns
        -------
        img: image after ROI interchange
        """
        mode = self.P2Pmode
        size = self.P2Ppatch
        if mode == 0:
            return imga

        if len(imgb) == 0:
            mode = 2    
        if mode == 1: #interchange along t-axial 
            img = self.interchange_multiple3d(imga, imgb, ifdirect = True)
        elif mode == 2: #interchange in single image
            img = self.interchange_single3d(imga)
        elif mode == 3: #interchange in different images
            img = self.interchange_multiple3d(imga, imgb, ifdirect = False)

        return img

    def interchange_multiple3d(self, imga, imgb, ifdirect = False):
        """
        SN2N tool: Core of random interchange in multiple images
        ------
        imga
            image TO ROI interchange
        imgb
            another image for ROI interchange
            {default: []}
        ifdirect
            interchange in same (True) or different (False) ROIs
            {default: False}

        Returns
        -------
        imga: image after ROI interchange
        """
        size = self.P2Ppatch
        h = size[0]
        w = size[1]
        if h < np.min((np.size(imga, 1), np.size(imgb, 1))) and w < np.min((np.size(imga, 2), np.size(imgb, 2))):
            xa = random.randint(0, np.size(imga, 1) - h)
            ya = random.randint(0, np.size(imga, 2) - w)
            if ifdirect:
                xb = xa
                yb = ya
                imga[:, xa: xa + h, ya : ya + w] = imgb[:, xa: xa + h, ya : ya + w] 
            else:
                xb = random.randint(0, np.size(imgb, 1) - h)
                yb = random.randint(0, np.size(imgb, 2) - w)
                imga[:, xa: xa + h, ya : ya + w] = imgb[:, xb : xb + h, yb : yb + w]

        return imga

    def interchange_single3d(self, img):
        """
        SN2N tool: Core of random interchange in single image
        ------
        img
            image TO ROI interchange

        Returns
        -------
        img: image after ROI interchange
        """
        size = self.P2Ppatch
        h = size[0]
        w = size[1]

        if h < np.size(img, 1) and w < np.size(img, 2):
            x1 = random.randint(0, np.size(img, 1) - h)
            y1 = random.randint(0, np.size(img, 2) - w)
            x2 = random.randint(0, np.size(img, 1) - h)
            y2 = random.randint(0, np.size(img, 2) - w)
            img[:, x1: x1 + h, y1 : y1 + w], img[:, x2 : x2 + h, y2 : y2 + w] = \
            img[:, x2 : x2 + h, y2 : y2 + w], img[:, x1: x1 + h, y1 : y1 + w]
        return img



