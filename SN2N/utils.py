# -*- coding: utf-8 -*-

import numpy as np
import tifffile
import torch
import pywt

def imread_stack(self, imgpath):
    image_stack = tifffile.imread(imgpath)
    return image_stack

def normalize(stack):
    stack = stack.astype('float32')
    stack = stack - np.min(stack)
    stack = stack / np.max(stack)
    return stack

def normalize_tanh(input):
    img = np.array(input).astype('float32')
    img = img * 2 + 1
    img = img - np.min(img)
    img = img / np.max(img)
    return img

def TOTENSOR_(img_datas_np):
    if len(img_datas_np.shape) == 2:
       img_datas_np = np.expand_dims(img_datas_np, axis = 0)
       img_datas_np = np.expand_dims(img_datas_np, axis = 0)
    return torch.from_numpy(img_datas_np).type(torch.FloatTensor)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32): 
     if dtype is not None: 
         x = x.astype(dtype,copy=False) 
         mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False) 
         ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False) 
         eps = dtype(eps) 
     try: 
         import numexpr 
         x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )") 
     except ImportError: 
         x = (x - mi) / ( ma - mi + eps ) 
     if clip: 
        x = np.clip(x,0,1) 
        return x 

def normalize_percentage(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32): 
    """Percentile-based image normalization.""" 
    mi = np.percentile(x,pmin,axis=axis,keepdims=True) #np.percantile对象可以是多维数组
    ma = np.percentile(x,pmax,axis=axis,keepdims=True) 
    
    # print(np.double(mi), ma)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype) 


def Low_frequency_resolve(coeffs, dlevel):
    '''    
    Extracts and processes the low-frequency components from the wavelet coefficients.
    -------    
    coeffs (tuple)
        A tuple containing the coefficients from wavelet decomposition.
    dlevel (int)
        The decomposition level at which the wavelet transform was performed.
    -------
    Returns:
        list: A list of modified wavelet coefficients with high-frequency components set to zero.
    '''
    cAn = coeffs[0]
    vec = []
    vec.append(cAn)
    for i in range(1, dlevel+1):
        (cH, cV, cD) = coeffs[i]
        [cH_x, cH_y] = cH.shape
        cH_new = np.zeros((cH_x, cH_y))
        t = (cH_new, cH_new, cH_new)
        vec.append(t)
    return vec


def rm_1(Biter, x, y):
    '''
    Extracts and processes the low-frequency components from the wavelet coefficients.
    -------    
    coeffs (tuple)
        A tuple containing the coefficients from wavelet decomposition.
    dlevel (int)
        The decomposition level at which the wavelet transform was performed.
    -------
    Returns:
        list: A list of modified wavelet coefficients with high-frequency components set to zero.
    '''
    Biter_new = np.zeros((x, y), dtype=('uint8'))
    if x%2 and y%2 == 0:
        Biter_new[:, :] = Biter[0:x, :]
    elif x%2 == 0 and y%2:
        Biter_new[:, :]  = Biter[:, 0:y]
    elif x%2 and y%2:
        Biter_new[:, :]  = Biter[0:x, 0:y]
    else:
        Biter_new = Biter
    return Biter_new


def background_estimation_stack(imgs, th = 1, dlevel = 7, wavename = 'db6', iter = 3):
    ''' 
    Background estimation
    function Background = background_estimation(imgs,th,dlevel,wavename,iter)
    imgs: ndarray
        Input image (can be N dimensional).
    th : int, optional
        if iteration {default:1}
    dlevel : int, optional
     decomposition level {default:7}
    wavename
     The selected wavelet function {default:'db6'}
    iter:  int, optional
     iteration {default:3}
    -----------------------------------------------
    Return:
     Background
    '''
    try:
        [t, x, y] = imgs.shape 
        Background = np.zeros((t, x, y))
        for taxial in range(t):
            img = imgs[taxial, :, :]
            for i in range(iter):
                initial = img
                res = initial
                coeffs = pywt.wavedec2(res, wavelet = wavename, level = dlevel)
                vec = Low_frequency_resolve(coeffs, dlevel)
                Biter = pywt.waverec2(vec, wavelet = wavename)
                Biter_new = rm_1(Biter, x, y)
                if th > 0:
                    eps = np.sqrt(np.abs(res))/2
                    ind = initial>(Biter_new+eps)
                    res[ind] = Biter_new[ind]+eps[ind]
                    coeffs1 = pywt.wavedec2(res, wavelet = wavename, level = dlevel)
                    vec = Low_frequency_resolve(coeffs1, dlevel)
                    Biter =  pywt.waverec2(vec, wavelet = wavename)
                    Biter_new = rm_1(Biter, x, y)
                    Background[taxial, :, :] = Biter_new
    except ValueError:
        [x, y] = imgs.shape 
        Background = np.zeros((x, y))
        for i in range(iter):
            initial = imgs
            res = initial
            coeffs = pywt.wavedec2(res, wavelet = wavename, level = dlevel)
            vec = Low_frequency_resolve(coeffs, dlevel)
            Biter = pywt.waverec2(vec, wavelet = wavename)
            Biter_new = rm_1(Biter, x, y)
            if th > 0:
                eps = np.sqrt(np.abs(res))/2
                ind = initial>(Biter_new+eps)
                res[ind] = Biter_new[ind]+eps[ind]
                coeffs1 = pywt.wavedec2(res, wavelet = wavename, level = dlevel)
                vec = Low_frequency_resolve(coeffs1, dlevel)
                Biter =  pywt.waverec2(vec, wavelet = wavename)
                Biter_new = rm_1(Biter, x, y)
                Background = Biter_new
    return Background