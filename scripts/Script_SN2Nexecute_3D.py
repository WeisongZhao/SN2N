# -*- coding: utf-8 -*-

import sys
from SN2N.get_options import execute3D
from SN2N.SN2Nexecute import SN2Nexecute_3D

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
        
    ======Other parameters do not require modification; for details, refer to SN2N.get_options.========
    
    """
    
    img_path = 'C:/Users/qqq/Desktop/SN2N-V0.3.0/examples/denoising3D/data/raw_data'
    P2Pmode = '1'
    P2Pup = '1'
    BAmode = '1'
    SWsize = '64' 
    sn2n_loss = '1'
    bs = '4'
    lr = '2e-4'
    epochs = '1'
    model_path = 'C:/Users/qqq/Desktop/SN2N-V0.3.0/examples/denoising3D/data/models'
    infer_mode = '0'
    overlap_shape = '2,256,256'
    
    
    execute3D_args = [
        '--img_path', img_path,
        '--P2Pmode', P2Pmode,
        '--P2Pup', P2Pup,
        '--BAmode', BAmode,
        '--SWsize', SWsize,
        '--sn2n_loss', sn2n_loss,
        '--bs', bs,
        '--lr', lr,
        '--epochs', epochs,
        '--model_path', model_path,
        '--infer_mode', infer_mode,
        '--overlap_shape', overlap_shape
    ]

    if len(sys.argv) > 1:
        args = execute3D()
    else:
        args = execute3D(execute3D_args)
    
    print("Parsed arguments:", args) 
    
    ##Step 2: Executing.  
    SN2Nexecute_3D(args)
    
    
   
