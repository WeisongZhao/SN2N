# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pywt
from skimage.io import imread, imsave, imshow

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






