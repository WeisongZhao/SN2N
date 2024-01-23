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
        

    def imread_stack(self, imgpath):
        image_stack = tifffile.imread(imgpath)
        return image_stack
    
    def block_multiframe(self, image_stack):
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
    
    def slidingWindow_multiframe(self, image_data_stack, threshold_mode = 1, threshold = 15):
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
        flage = 0
        (t, h, w) = image_data_stack.shape
        interval = self.sliding_interval
        image_data_stack = 255*self.normalize(image_data_stack)
        if threshold_mode == 1:
            threshold_real = threshold
        if threshold_mode == 2:
            avg_list = []
            for taxial in range(t):
                img = image_data_stack[taxial, :, :]
                avg = np.mean(img)
                avg_list.append(avg)
            avg_list = np.array(avg_list)
            avg = np.mean(avg_list)
            threshold_real = avg+threshold
        bsize = self.img_res[0]
        
        
        xx = int(np.floor(h - (bsize - interval)) / interval)
        yy = int(np.floor(w - (bsize - interval)) / interval)
        
        image_arr = []
        for i in range(1, (xx + 1)):
            for j in range(1, (yy + 1)):
                left1 = (j - 1) * interval
                right1 = (j - 1) * interval + bsize
                down = (i - 1) * interval
                up = (i - 1) * interval + bsize 
                img = image_data_stack[:, down:up, left1:right1]  
                if np.sum(img[taxial]) > bsize * bsize * (threshold_real):
                    image_arr.append(img)
        image_arr = np.array(image_arr)
        return image_arr
    
    
    def slidingWindow_multiframe_V2(self, image_data_stack, threshold_mode = 1, threshold = 15):
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
        flage = 0
        # 16,x,y
        (t, h, w) = image_data_stack.shape
        interval = self.sliding_interval
        image_data_stack = 255*self.normalize(image_data_stack)
        if threshold_mode == 1:
            threshold_real = threshold
        if threshold_mode == 2:
            avg_list = []
            for taxial in range(t):
                img = image_data_stack[taxial, :, :]
                avg = np.mean(img)
                avg_list.append(avg)
        
        bsize = self.img_res[0]
        
        
        xx = int(np.floor(h - (bsize - interval)) / interval)
        yy = int(np.floor(w - (bsize - interval)) / interval)
        
        image_arr_raw = []
        image_arr = []
        for i in range(1, (xx + 1)):
            for j in range(1, (yy + 1)):
                left1 = (j - 1) * interval
                right1 = (j - 1) * interval + bsize
                down = (i - 1) * interval
                up = (i - 1) * interval + bsize 
                img = image_data_stack[:, down:up, left1:right1]  
                image_arr_raw.append(img)
        image_arr_raw = np.array(image_arr_raw)
        [pp, tt, xx, yy] = image_arr_raw.shape
        print(image_arr_raw.shape)
        
        # 352,16,128,128
        for ppp in range(pp):
            flage = 0
            for ttt in range(tt): 
                img_temp = image_arr_raw[ppp, ttt, :, :]
                threshold_real = avg_list[ttt] + threshold
                avg_list.append(avg)
                if np.sum(img_temp) > bsize * bsize * (threshold_real):
                    flage = flage + 1
            print(flage)
            if flage==t:
                image_arr.append(image_arr_raw[ppp, :, :, :])
        image_arr = np.array(image_arr)
        print(image_arr.shape)
        image_arr = np.squeeze(image_arr)
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
        Fourier re-scale
        ------
        image_stack
            image TO Fourier interpolation
        
        Returns
        -------
        imgf1: image with img_res size 
        """
    
        imsize = self.img_res
        [xx, yy] = imsize
        
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
                
            n = np.array([(xx/x), (yy/y)])
            
            padsize = np.array([(xx-x)/2, (yy-y)/2], dtype = 'int')
            pad_hei = np.ceil(padsize[0]).astype('int')
            pad_wid = np.ceil(padsize[1]).astype('int')
            
            img = np.pad(img, ((pad_hei, 0), (pad_hei, 0)), 'symmetric')
            img = np.pad(img, ((0, pad_wid), (0, pad_wid)),  'symmetric')
            
            tttem1 = np.multiply(n, imsize)
            tttem2 = np.subtract(n, 1)
            newsz = np.array((np.round(np.subtract(tttem1, tttem2))).astype('int'))
            
            img1 = self.interpft(img, newsz[0], 0)
            img1 = self.interpft(img1, newsz[1], 1)
            
            imgsz_big = np.array(img1.shape)
            temmm = np.array([2, 2])
            idx1 = np.array(np.divide((np.subtract(imgsz_big, imsize)), temmm)).astype('int')
            
            ttttem1 = np.subtract(np.multiply(n[0], imgsz[0]), 1).astype('int')
            ttttem2 = np.subtract(np.multiply(n[1], imgsz[1]), 1).astype('int')
            
            
            imgf1[slice, :, :] = img1[idx1[0] : idx1[0] + ttttem1 + 1, idx1[1] :idx1[1]+ ttttem2 + 1]
            imgf1[imgf1 < 0] = 0
        return imgf1
    
    
    def fourier_inter_multiframe(self, image_stack):
        """
        Fourier re-scale
        ------
        image_stack
            image TO Fourier interpolation
        
        Returns
        -------
        imgf1: image with img_res size 
        """
    
        imsize = self.img_res
        [xx, yy] = imsize
        
        if image_stack.ndim == 2:        
            image_stack = np.expand_dims(image_stack, 0)
        [frame, t, x, y] = image_stack.shape
        imgf1 = np.zeros((frame, t, imsize[0], imsize[1]))
        
        
        for f in range(frame):
            for slice in range(t):
                img = image_stack[f, slice, :, :]
                imgsz = np.array([x, y])
                tem1 = np.divide(imgsz, 2)
                tem2 = np.multiply(tem1, 2)
                tem3 = np.subtract(imgsz, tem2)
                b = (tem3 == np.array([0, 0]))
                if b[0] == True:
                    sz = imgsz - 1
                else:
                    sz = imgsz     
                    
                n = np.array([(xx/x), (yy/y)])
                
                padsize = np.array([(xx-x)/2, (yy-y)/2], dtype = 'int')
                pad_hei = np.ceil(padsize[0]).astype('int')
                pad_wid = np.ceil(padsize[1]).astype('int')
                
                img = np.pad(img, ((pad_hei, 0), (pad_hei, 0)), 'symmetric')
                img = np.pad(img, ((0, pad_wid), (0, pad_wid)),  'symmetric')
                
                tttem1 = np.multiply(n, imsize)
                tttem2 = np.subtract(n, 1)
                newsz = np.array((np.round(np.subtract(tttem1, tttem2))).astype('int'))
                
                img1 = self.interpft(img, newsz[0], 0)
                img1 = self.interpft(img1, newsz[1], 1)
                
                imgsz_big = np.array(img1.shape)
                temmm = np.array([2, 2])
                idx1 = np.array(np.divide((np.subtract(imgsz_big, imsize)), temmm)).astype('int')
                
                ttttem1 = np.subtract(np.multiply(n[0], imgsz[0]), 1).astype('int')
                ttttem2 = np.subtract(np.multiply(n[1], imgsz[1]), 1).astype('int')
                
                
                imgf1[f, slice, :, :] = img1[idx1[0] : idx1[0] + ttttem1 + 1, idx1[1] :idx1[1]+ ttttem2 + 1]
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
    
    def savedata_multiframe(self, image_stack, flage):
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
        
        left, right = self.block_multiframe(image_stack) 
        if left.ndim==3:
            return flage
            
        [frame, t, x, y] = left.shape
        if self.ifx2:
            imsize = self.img_res
            if self.inter_method == 'bilinear':
                left = self.imgstack_resize(left, imsize)
                right = self.imgstack_resize(right, imsize)
            elif self.inter_method == 'Fourier':
                
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
                # print(left_stack.shape)
        else:
            imsize = self.img_res
            imsize = (int(imsize[0] / 2), int(imsize[1] / 2))
        [frame, t, x, y] = left_stack.shape
        size1 = (t, imsize[0], imsize[1] * 2)
        imgpart = np.zeros(size1, dtype = 'float32')
        imgpart_aug = np.zeros(size1, dtype = 'float32')
        imgpart_list = []
        for f in range(frame):
            temp_l = left_stack[f, :, :, :]
            temp_l = self.normalize(temp_l)
            temp_r = right_stack[f, :, :, :]
            temp_r = self.normalize(temp_r)
            imgpart[:, 0 : imsize[0], 0 : imsize[1]] = temp_l        
            imgpart[:, 0 : imsize[0], imsize[1] : 2 * imsize[1]] = temp_r
            imgpart_copy = imgpart.copy()
            imgpart_list.append(imgpart_copy)
            if self.augment_mode == 1:
                mode = random.randint(0, 7)   
                for taxial in range(t):
                    temp_l_aug = self.data_augment(temp_l[taxial, :, :], mode)
                    temp_r_aug = self.data_augment(temp_r[taxial, :, :], mode)
                    imgpart_aug[taxial, 0 : imsize[0], 0 : imsize[1]] = temp_l_aug        
                    imgpart_aug[taxial, 0 : imsize[0], imsize[1] : 2 * imsize[1]] = temp_r_aug
                    imgpart_aug_copy = imgpart_aug.copy()
                    imgpart_list.append(imgpart_aug_copy)
            elif self.augment_mode == 2:
                for m in range(1, 8):
                    for taxial in range(t):
                        temp_l_aug = self.data_augment(temp_l[taxial, :, :], m)
                        temp_r_aug = self.data_augment(temp_r[taxial, :, :], m)
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
                imsave(('%s/%d.tif') %(self.save_path, flage), img.astype('uint8'))
                flage = flage + 1
                if flage % 100 == 0:
                    print('Saving training images:', flage)
        return flage

    
    def savedata4folder_agument_multiframe(self, multi_frame = 16, flage = 1, interval = 64, threshold_mode = 2, threshold = 15, 
                                                    size =(64, 64), times = 1, roll = 1):
        """
        SelfN2N tool: Generator with random interchange 
        ------
        flage
            started file index
            {default: 1}
        interval
            interval pixel number to slide
            {default: 64}
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
            image_data_path = datapath_list[ll : ll+multi_frame]
            print('For number %d frame'%(ll + 1))

            
            image_data_stack = []
            for f in range(multi_frame):
                img = self.imread_stack(image_data_path[f])
                image_data_stack.append(img)
            image_data_stack = np.array(image_data_stack)
            
            [t, x, y] = image_data_stack.shape
            image_arr = self.slidingWindow_multiframe(image_data_stack, 
                threshold_mode = threshold_mode, threshold = threshold)
            flage = self.savedata_multiframe(image_arr, flage)
            
            # image_data_stack[multiframe, x, y]
            image_aug_pre = []
            for circlelarge in range(roll):
                if times >= 1:
                    if self.pre_augment_mode == 3:
                            image_data_b = self.imread_stack(datapath_list[random.randint(0, l - 1)])
                    else:
                        image_data_b = image_data_stack[random.randint(0, t - 1),:,:]
                        
                    image_arr_temp = []
                    for taxial in range(t):
                        image_data = image_data_stack[taxial, :, :]
                        image_data_pre = self.random_interchange(imga = image_data,
                            imgb = image_data_b, size = size , mode = self.pre_augment_mode)   
                        
                        image_arr_temp.append(image_data_pre)
                    image_arr_temp = np.array(image_arr_temp)
                    image_aug_pre.append(image_arr_temp)
                        
                    #repeat agument N-1 times
                    for circle in range(times - 1):
                        if self.pre_augment_mode == 3:
                                image_data_b = self.imread_stack(datapath_list[random.randint(0, l - 1)])
                        else:
                            image_data_b = image_data_stack[random.randint(0, t - 1),:,:]
                            
                        image_arr_temp_temp = []
                        for taxial in range(t):
                            image_data_pre = image_arr_temp[taxial, :, :]    
                            
                            image_data_pre = self.random_interchange(imga = image_data_pre,
                                imgb = image_data_b, size = size , mode = self.pre_augment_mode)
                            
                            image_arr_temp_temp.append(image_data_pre)
                        image_arr_temp_temp = np.array(image_arr_temp)
                        image_aug_pre.append(image_arr_temp_temp)
                        
                    l = len(image_aug_pre)
                    for ll in range(l):
                        image_arr = self.slidingWindow_multiframe(image_aug_pre[ll], 
                                 threshold_mode = threshold_mode, threshold = threshold)
                        flage = self.savedata_multiframe(image_arr, flage)

            
        return   




    def savedata4folder_agument_multiframe_stack(self, multi_frame = 16, flage = 1, interval = 64, threshold_mode = 2, threshold = 15, 
                                                    size =(64, 64), times = 1, roll = 1):
        """
        SelfN2N tool: Generator with random interchange 
        ------
        flage
            started file index
            {default: 1}
        interval
            interval pixel number to slide
            {default: 64}
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
        l= len(datapath_list)
        for ll in range(l):
            image_data_stack = self.imread_stack(datapath_list[ll])
            
            image_data_stack_list = []
            [t, x, y] = image_data_stack.shape
            for tt in range(t-multi_frame):
                image_data = image_data_stack[tt : tt + multi_frame, :, :]
                image_data_stack_list.append(image_data)
            image_data_stack_list = np.array(image_data_stack_list)
            # print(image_data_stack_list.shape)
            [tt, xx, yy] = image_data_stack.shape
            
            
            
            for ttt in range(tt-16):
                img_temp = np.squeeze(image_data_stack_list[ttt, :, :, :])
                image_arr = self.slidingWindow_multiframe(img_temp, 
                    threshold_mode = threshold_mode, threshold = threshold)
                flage = self.savedata_multiframe(image_arr, flage)
            
        
    
            
        return 
        
        



