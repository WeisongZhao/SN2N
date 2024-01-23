# -*- coding: utf-8 -*-

import os
import scipy.misc
import numpy as np
import tifffile
from glob import glob
import skimage.io as io
from skimage.io import imsave, imread, imshow
import skimage.transform as st
import random
np.seterr(divide='ignore',invalid='ignore')


class data_generator():
    def __init__(self, img_path, save_path, pre_augment_mode = 0, augment_mode = 0, 
                    img_res = (128, 128), ifx2 = True, inter_method = 'Fourier', sliding_interval = 64):
        """
        SelfN2N data generator
        ------
        img_path
            path of raw images TO train
        save_path
            path of patched dataset TO save
        pre_augment_mode
            0: NONE; 
            1: direct interchange in t;
            2: interchange in single frame;
            3: interchange in multiple frame but in different regions
            {default: 0}
        augment_mode
            0: NONE; 
            1: double the dataset with random rotate&flip;
            2: eightfold the dataset with random rotate&flip;
            {default: 0}
        img_res
            patch size
            {default: (128, 128)}
        ifx2
            if re-scale TO original size
            True OR False
            {default: True}
        inter_method
            Scaling method
            'Fourier': Fourier re-scaling;
            'bilinear': spatial re-scaling;
            {default: 'Fourier'}
        
		------
        Example
        	DG = data_generator(img_path, save_path)
        	DG.savedata4folder_agument(threshold = 40)
        """
        self.pre_augment_mode = pre_augment_mode
        self.augment_mode = augment_mode
        self.img_path = img_path
        self.save_path = save_path
        self.img_res = img_res
        self.ifx2 = ifx2
        self.inter_method = inter_method
        self.sliding_interval = sliding_interval
        
    def imgread_legacy(self, imgpath):
        """
        Not in use
        """
        img_stack = []
        img = imread(imgpath)
        img_stack.append(img)
        return img_stack

    def imread(self, imgpath):
        """
        Not in use
        """
        return scipy.misc.imread(imgpath).astype(np.float)
    
    def imread_stack(self, imgpath):
        image_stack = tifffile.imread(imgpath)
        return image_stack
    
    def block(self, image_stack):
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
            imsize = self.img_res
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
    
    def slidingWindow(self, image_data, threshold_mode = 1, threshold = 15):
        """
        SelfN2N tool: patch
        ------
        image_data
            image TO generate
        interval
            interval pixel number to slide
            {default: 64}
        threshold (0 ~ 255)
            threshold to exclude some black patches
            {default: 15}
        
        Returns
        -------
        image_arr: patches with size (self.img_res)
        """
        interval = self.sliding_interval
        image_data = 255*self.normalize(image_data)
        if threshold_mode == 1:
            threshold_real = threshold
        if threshold_mode == 2:
            avg = np.mean(image_data)
            threshold_real = avg+threshold
        bsize = self.img_res[0]
        image_arr = []
        (h, w) = image_data.shape
        xx = int(np.floor(h - (bsize - interval)) / interval)
        yy = int(np.floor(w - (bsize - interval)) / interval)
        for i in range(1, (xx + 1)):
            for j in range(1, (yy + 1)):
                left1 = (j - 1) * interval
                right1 = (j - 1) * interval + bsize
                down = (i - 1) * interval
                up = (i - 1) * interval + bsize                    
                img = image_data[down:up, left1:right1]  
                if np.sum(img) > bsize * bsize * (threshold_real):
                    image_arr.append(img)
                # if np.sum(img) > bsize * bsize * threshold:
                #     image_arr.append(img)
        image_arr = np.array(image_arr)
        return image_arr
    
    def imwrite(self, image_path, image):
        """
        Not in use
        """
        image = image.astype('float32')
        image = image - np.min(image)
        image = 255 * image / np.max(image)
        img2save = image.astype('uint8')              
        imsave(image_path, img2save)
        

    
    def fourier_inter(self, image_stack):
        """
        SelfN2N tool: Fourier re-scale
        ------
        image_stack
            image TO Fourier interpolation
        
        Returns
        -------
        imgf1: image with 2x size 
        """
        imsize = self.img_res
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
    
    
    def data_augment(self, img_data, mode):    
        """
        SelfN2N tool: Random flip&rotate
        ------
        img_data
            image TO augmentation
        mode
            mode of flip&rotate

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
    

    def random_interchange(self, imga, imgb = [], size = (64, 64), mode = 1):
        """
        SelfN2N tool: Random interchange
        ------
        imga
            image TO ROI interchange
        imgb
            another image for ROI interchange
            {default: []}
        size
            ROI size
            {default: (64, 64)}
        mode
            1: direct interchange (same ROI) in t axial
            2: interchange in single image
            3: interchange (two ROIs) in multiple images

        Returns
        -------
        img: image after ROI interchange
        """
        if mode == 0:
            return imga

        if imgb == []:
            mode = 2    
        if mode == 1: #interchange along t-axial 
            img = self.interchange_multiple(imga, imgb, size = size, ifdirect = False)
        elif mode == 2: #interchange in single image
            img = self.interchange_single(imga, size = size)
        elif mode == 3: #interchange in different images
            img = self.interchange_multiple(imga, imgb, size = size, ifdirect = False)

        return img

    def interchange_multiple(self, imga, imgb, size = (64, 64), ifdirect = False):
        """
        SelfN2N tool: Core of random interchange in multiple images
        ------
        imga
            image TO ROI interchange
        imgb
            another image for ROI interchange
            {default: []}
        size
            ROI size
            {default: (64, 64)}
        ifdirect
            interchange in same (True) or different (False) ROIs
            {default: False}

        Returns
        -------
        imga: image after ROI interchange
        """
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

    def interchange_single(self, img, size = (64, 64)):
        """
        SelfN2N tool: Core of random interchange in single image
        ------
        img
            image TO ROI interchange
        size
            ROI size
            {default: (64, 64)}

        Returns
        -------
        img: image after ROI interchange
        """
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

    def normalize(self, stack):
        stack = stack.astype('float32')
        stack = stack - np.min(stack)
        stack = stack / np.max(stack)
        return stack
    
    def savedata(self, image_stack, flage):
        """
        SelfN2N tool: TO save data
            DATA structure: img, label (h, 2 x h) uint8
        ------
        image_stack
            data TO save
        flage  
            data number

        Returns
        -------
        NULL
        """
        
        left, right = self.block(image_stack)            
        if self.ifx2:
            imsize = self.img_res
            if self.inter_method == 'bilinear':
                left = self.imgstack_resize(left, imsize)
                right = self.imgstack_resize(right, imsize)
            elif self.inter_method == 'Fourier':
                left = self.fourier_inter(left)
                right = self.fourier_inter(right)
        else:
            imsize = self.img_res
            imsize = (int(imsize[0] / 2), int(imsize[1] / 2))
        [t, x, y] = left.shape
        size1 = (imsize[0], imsize[1] * 2)
        imgpart = np.zeros(size1, dtype = 'float32')
        imgpart_aug = np.zeros(size1, dtype = 'float32')
        imgpart_list = []
        for tt in range(t):
            temp_l = left[tt, :, :]
            temp_l = self.normalize(temp_l)
            temp_r = right[tt, :, :]
            temp_r = self.normalize(temp_r)
            imgpart[0 : imsize[0], 0 : imsize[1]] = temp_l        
            imgpart[0 : imsize[0], imsize[1] : 2 * imsize[1]] = temp_r
            imgpart_copy = imgpart.copy()
            imgpart_list.append(imgpart_copy)
            if self.augment_mode == 1:
                mode = random.randint(0, 7)                
                temp_l_aug = self.data_augment(temp_l, mode)
                temp_r_aug = self.data_augment(temp_r, mode)
                imgpart_aug[0 : imsize[0], 0 : imsize[1]] = temp_l_aug        
                imgpart_aug[0 : imsize[0], imsize[1] : 2 * imsize[1]] = temp_r_aug
                imgpart_aug_copy = imgpart_aug.copy()
                imgpart_list.append(imgpart_aug_copy)
            elif self.augment_mode == 2:
                for m in range(1, 8):
                    temp_l_aug = self.data_augment(temp_l, m)
                    temp_r_aug = self.data_augment(temp_r, m)
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
                imsave(('%s/%d.tif') %(self.save_path, flage), img.astype('uint8'))
                flage = flage + 1
                if flage % 100 == 0:
                    print('Saving training images:', flage)
        return flage

    def savedata4folder_agument(self, flage = 1, threshold_mode = 2, threshold = 15, 
                                                    size =(64, 64), times = 1, roll = 1):
        """
        SelfN2N tool: Generator with random interchange 
        ------
        flage
            started file index
            {default: 1}
        threshold_mode (0 ~ 2)
            threshold mode to exclude some black patches
            {default: 2}
        threshold (0 ~ 255)
            threshold to exclude some black patches
            {default: 15}
        size
            ROI size of interchange
            {default: (64, 64)}
        times
            Repeat times of ROI interchange
        roll
            Repeat times of interchange for one image

        Returns
        -------
        NULL
        """
        if self.pre_augment_mode == 0:
            times = 0
            roll = 0
        datapath_list = []
        for (root, dirs, files) in os.walk(self.img_path):
            for j, Ufile in enumerate(files):
                img_path = os.path.join(root, Ufile)
                datapath_list.append(img_path)
                  
        l = len(datapath_list)

        for ll in range(l):
            image_data = self.imread_stack(datapath_list[ll])
            print('For number %d frame'%(ll + 1))

            try:
                [t, x, y] = image_data.shape
                for taxial in range(t):
                    #original data save
                    image_arr = self.slidingWindow(image_data[taxial,:,:], 
                        threshold_mode = threshold_mode, threshold = threshold)
                    flage = self.savedata(image_arr, flage)

                    for circlelarge in range(roll):
                        if times >= 1:
                            image_data_pre = self.random_interchange(imga = image_data[taxial,:,:],
                                imgb = image_data[random.randint(0, t - 1),:,:], 
                                    size = size , mode = self.pre_augment_mode)                        
                            #repeat agument N-1 times
                            for circle in range(times - 1):
                                image_data_pre = self.random_interchange(imga = image_data_pre,
                                    imgb = image_data[random.randint(0, t - 1),:,:], 
                                        size = size , mode = self.pre_augment_mode)
                            image_arr = self.slidingWindow(image_data_pre, 
                                    threshold_mode = threshold_mode, threshold = threshold)
                            flage = self.savedata(image_arr, flage)

            except ValueError:
                if self.pre_augment_mode == 3:
                    image_data_b = self.imread_stack(datapath_list[random.randint(0, l - 1)])
                else:
                    image_data_b = []
                #original data save
                image_arr = self.slidingWindow(image_data, threshold_mode = threshold_mode, threshold = threshold)
                flage = self.savedata(image_arr, flage)
                
                for circlelarge in range(roll):
                    if times >= 1:
                        image_data_pre = self.random_interchange(imga = image_data,
                            imgb = image_data_b, size = size , mode = self.pre_augment_mode) 
                        #repeat agument N-1 times
                        for circle in range(times - 1):
                            image_data_pre = self.random_interchange(imga = image_data_pre,
                                imgb = image_data_b, size = size , mode = self.pre_augment_mode)
                        image_arr = self.slidingWindow(image_data_pre, 
                                threshold_mode = threshold_mode, threshold = threshold)
                        flage = self.savedata(image_arr, flage)
        return   

        
        



