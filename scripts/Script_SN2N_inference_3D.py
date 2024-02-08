# -*- coding: utf-8 -*-

import sys
from SN2N.inference import Predictor3D
from SN2N.get_options import Predict3D
    

if __name__ == '__main__':
    ##Step 1: Define custom parameters.
    """
    -----Parameters------
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
    
    img_path = 'C:/Users/qqq/Desktop/SN2N-V0.3.0/examples/denoising3D/data/raw_data'
    model_path = 'C:/Users/qqq/Desktop/SN2N-V0.3.0/examples/denoising3D/data/models'
    infer_mode = '1'
    overlap_shape = '2,256,256'
    
    Predict3D_args = [
        '--img_path', img_path,
        '--model_path', model_path,
        '--infer_mode', infer_mode,
        '--overlap_shape', overlap_shape
    ]
    
    if len(sys.argv) > 1:
        args = Predict3D()
    else:
        args = Predict3D(Predict3D_args)
    
    print("Parsed arguments:", args)   
    
    ##Step 2: Execute predicting.
    p = Predictor3D(img_path = args.img_path, model_path = args.model_path, infer_mode = args.infer_mode, overlap_shape = args.overlap_shape)
    p.execute()
