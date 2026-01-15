# -*- coding: utf-8 -*-

import os
import torch
import tifffile
import numpy as np
import itertools
import tqdm
from SN2N.utils import *

class Predictor2D(): 
    def __init__(self, img_path, model_path, infer_mode):
        """
        Self-inspired Noise2Noise
        
        -----Parameters------
        =====Important==========
        img_path:
            Path of raw images to inference
        model_path:
            Path of model for inference
        infer_mode:
            Prediction Mode
            0: Predict the results of all models generated during training 
            under the default "models" directory on the img_path.                
            1: Predict the results of the models provided by the user under 
            the given model_path on the Img_path provided by the user.
            
        """
        self.img_path = img_path
        self.parent_dir = os.path.dirname(img_path)
        self.dataset_path = os.path.join(self.parent_dir, 'datasets')
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        self.model_save_path = os.path.join(self.parent_dir, 'models')
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        self.save_path = os.path.join(self.parent_dir, 'predictions')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.model_path = model_path
        self.infer_mode = infer_mode
        
    def execute(self):
        img_path = self.img_path
        infer_mode = self.infer_mode
        if infer_mode == 0:
            model_path = self.model_save_path
        else:
            model_path = self.model_path
        save_path = self.save_path
        print('The path for the raw images used for training is located under:\n%s' %(img_path))
        print('Models is being saved under:\n%s' %(model_path))
        print('Predictions is being saved under:\n%s' %(save_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for (mroot, mdirs, mfiles) in os.walk(model_path):
            for jj, model_Ufile in enumerate(mfiles):
                print('=====Model: %d====='%(jj + 1))
                m_path = os.path.join(mroot, model_Ufile)
                model = torch.load(m_path, map_location=device)
                model = model.to(device)
                with torch.no_grad():
                    for (root, dirs, files) in os.walk(img_path):
                        for j, Ufile in enumerate(files):
                            imgpath = os.path.join(root, Ufile)
                            image_data = tifffile.imread(imgpath)
                            try:
                                [t, x, y] = image_data.shape
                                test_pred_np = np.zeros((t,x,y))
                                for taxial in range(t):
                                    datatensor = TOTENSOR_(normalize(image_data[taxial,:,:]))
                                    test_pred = model(datatensor.to(device))
                                    test_pred = test_pred.to(torch.device("cpu"))
                                    test_pred_np[taxial,:,:] = 255 * normalize(test_pred.numpy())
                                    os.makedirs(save_path, exist_ok=True)
                                tifffile.imwrite('%s/%s_%s.tif'%(save_path, Ufile, model_Ufile),test_pred_np.astype('uint8'))
                                print('Frame: %d'%(j + 1))
                            except ValueError:
                                datatensor = TOTENSOR_(normalize(image_data))
                                test_pred = model(datatensor.to(device))
                                test_pred = test_pred.to(torch.device("cpu"))
                                test_pred_np = 255 * normalize(test_pred.numpy())
                                os.makedirs(save_path, exist_ok=True)
                                tifffile.imwrite('%s/%s_%s.tif'%(save_path, Ufile, model_Ufile),test_pred_np.astype('uint8'))
                                print('Frame: %d'%(j + 1))
            return 
                            
class Predictor3D(): 
    def __init__(self, img_path, model_path, infer_mode, overlap_shape='2,256,256'):
        """
        Self-inspired Noise2Noise
        
        -----Parameters------
        =====Important==========
        img_path:
            Path of raw images to inference
        model_path:
            Path of model for inference
        infer_mode:
            Prediction Mode
            0: Predict the results of all models generated during training 
            under the default "models" directory on the img_path.                
            1: Predict the results of the models provided by the user under 
            the given model_path on the Img_path provided by the user.
        overlap_shape:
            Overlap shape in 3D stitching prediction.
            {default: '2, 256, 256'}
        """
        self.img_path = img_path
        self.parent_dir = os.path.dirname(img_path)
        self.dataset_path = os.path.join(self.parent_dir, 'datasets')
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        self.model_save_path = os.path.join(self.parent_dir, 'models')
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        self.save_path = os.path.join(self.parent_dir, 'predictions')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.model_path = model_path
        self.infer_mode = infer_mode
        self.overlap_shape = tuple(map(int, overlap_shape.split(',')))
            
    
    def execute(self, verbose=False):
        '''
        Applies a model to an input image. The input image stack is split into
        sub-blocks with model's input size, then the model is applied block by
        block. The sizes of input and output images are assumed to be the same
        while they can have different numbers of channels.
    
        Parameters
        ----------
        model: keras.Model
            Keras model.
        data: array_like or list of array_like
            Input data. Either an image or a list of images.
        overlap_shape: tuple of int or None
            Overlap size between sub-blocks in each dimension. If not specified,
            a default size ((32, 32) for 2D and (2, 32, 32) for 3D) is used.
            Results at overlapped areas are blended together linearly.
    
        Returns
        -------
        ndarray
            Result image.
        '''
        img_path = self.img_path
        infer_mode = self.infer_mode
        if infer_mode == 0:
            model_path = self.model_save_path
        else:
            model_path = self.model_path
        save_path = self.save_path
        print('The path for the raw images used for training is located under:\n%s' %(img_path))
        print('Models is being saved under:\n%s' %(model_path))
        print('Predictions is being saved under:\n%s' %(save_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        overlap_shape = self.overlap_shape
        modelpath_list = []
        for (mroot, mdirs, mfiles) in os.walk(model_path):
            for jj, model_Ufile in enumerate(mfiles):
                print('=====Model: %d====='%(jj + 1))
                m_path = os.path.join(mroot, model_Ufile)
                model = torch.load(m_path, map_location=device)
                model = model.to(device)
                with torch.no_grad():
                    filenames = []
                    for (root, dirs, files) in os.walk(img_path):
                        for j, Ufile in enumerate(files):
                            imgpath = os.path.join(root, Ufile)
                            input_tif = tifffile.imread(imgpath)
    
                            data = np.squeeze(input_tif).astype('uint8')
                            model_input_image_shape = (16, 640, 640)
                            model_output_image_shape = (16, 640, 640)
                        
                            if len(model_input_image_shape) != len(model_output_image_shape):
                                raise NotImplementedError
                        
                            image_dim = len(model_input_image_shape)
                            num_input_channels = 1
                            num_output_channels = 1
                        
                        
                            if overlap_shape is None:
                                if image_dim == 2:
                                    overlap_shape = (32, 32)
                                elif image_dim == 3:
                                    overlap_shape = (2, 32, 32)
                                else:
                                    raise NotImplementedError
                            elif len(overlap_shape) != image_dim:
                                raise ValueError(f'Overlap shape must be {image_dim}D; '
                                                 f'Received shape: {overlap_shape}')
                        
                            step_shape = tuple(
                                m - o for m, o in zip(
                                    model_input_image_shape, overlap_shape))
                        
                            block_weight = np.ones(
                                [m - 2 * o for m, o
                                 in zip(model_output_image_shape,overlap_shape)],
                                dtype=np.float32)
                        
                            block_weight = np.pad(
                                block_weight,
                                [(o + 1, o + 1) for o in overlap_shape],
                                'linear_ramp'
                            )[(slice(1, -1),) * image_dim]
                        
                            batch_size = 1
                            batch = np.zeros(
                                (batch_size, num_input_channels, *model_input_image_shape),
                                dtype=np.float32)
                        
                            if isinstance(data, (list, tuple)):
                                input_is_list = True
                            else:
                                data = [data]
                                input_is_list = False
                        
                            result = []
                        
                            for image in data:
                                input_image_shape = image.shape
                                output_image_shape = input_image_shape
                        
                                applied = np.zeros(
                                    (output_image_shape), dtype=np.float32)
                                sum_weight = np.zeros(output_image_shape, dtype=np.float32)
                        
                                num_steps = tuple(
                                    i // s + (i % s != 0)
                                    for i, s in zip(input_image_shape, step_shape))
                        
                                # top-left corner of each block
                                blocks = list(itertools.product(
                                    *[np.arange(n) * s for n, s in zip(num_steps, step_shape)]))
                        
                                for chunk_index in tqdm.trange(
                                        0, len(blocks), batch_size, disable=not verbose,
                                        dynamic_ncols=True, ascii=tqdm.utils.IS_WIN):
                                    rois = []
                                    for batch_index, tl in enumerate(
                                            blocks[chunk_index:chunk_index + batch_size]):
                                        br = [min(t + m, i) for t, m, i
                                              in zip(tl, model_input_image_shape, input_image_shape)]
                                        r1, r2 = zip(
                                            *[(slice(s, e), slice(0, e - s)) for s, e in zip(tl, br)])
                        
                                        m = image[r1]
                                        if model_input_image_shape != m.shape:
                                            pad_width = [(0, b - s)for b, s
                                                          in zip(model_input_image_shape, m.shape)]
                                            pad_width = np.array(pad_width, dtype = 'int')
                                            m = np.pad(m, pad_width, 'reflect')
                                            
                                            
                                            
                                        (t, x, y) = m.shape
                                        batch = np.zeros(
                                            (batch_size, num_input_channels, t, x, y),
                                            dtype=np.float32)
                                        batch[batch_index] = m
                                        rois.append((r1, r2))
                        
                                    p = np.zeros((batch.shape))
                                    datatensor = TOTENSOR_(batch)
                                    test_pred = model(datatensor.to(device))
                                    test_pred = test_pred.to(torch.device("cpu"))
                                    p[:, :, :, :, :] = test_pred.detach().numpy()
                                    
                                    
                                    
                                    for batch_index in range(len(rois)):
                                        for channel in range(num_output_channels):
                                            p[batch_index, channel, ...] *= block_weight
                        
                                        r1, r2 = [roi for roi in rois[batch_index]]
                                        applied[r1] += p[batch_index, channel][r2]
                                        sum_weight[r1] += block_weight[r2]
                        
                                for channel in range(num_output_channels):
                                    applied[...] /= sum_weight

                        
                                applied = 255*normalize(applied)
                                # 直接除以最大值进行归一化（不减去最小值）
                                #max_val = applied.max()
                                #if max_val > 0:  # 避免除以0
                                #    applied = 255 * (applied / max_val)
                                #else:
                                #    applied = 0  # 如果所有值都是0，则保持为0
                                result.append(applied)
                            r = result if input_is_list else result[0]
                            tifffile.imwrite('%s/%s_%s.tif'%(save_path, Ufile, model_Ufile),r.astype('uint8'))
                            print('Frame: %d'%(j + 1))

