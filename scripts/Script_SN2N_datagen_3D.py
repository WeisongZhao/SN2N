# -*- coding: utf-8 -*-

import os
import sys
current_script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_script_path, '..', 'SN2N'))
from datagen import generator3D
from get_options import datagen3D
    

if __name__ == '__main__':
    ##Step 1: Define custom parameters.
    """
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
        
    ======Other parameters do not require modification; for details, refer to SN2N.get_options.========
    """
    
    img_path = 'C:/Users/qqq/Desktop/SN2N-V0.28/examples/xyzt/data'
    P2Pmode = '1'
    P2Pup = '1'
    BAmode = '1'
    SWsize = '64' 
    
    
    datagen3D_args = [
        '--img_path', img_path,
        '--P2Pmode', P2Pmode,
        '--P2Pup', P2Pup,
        '--BAmode', BAmode,
        '--SWsize', SWsize        
    ]
    
    args = datagen3D(datagen3D_args)
    
    ##Step 2: Execute data generation.
    d = generator3D(img_path=args.img_path)
    d.execute()

