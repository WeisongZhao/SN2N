# -*- coding: utf-8 -*-

import sys
from SN2N.inference import Predictor2D
from SN2N.get_options import Predict2D
    

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
    """
    
    img_path = 'C:/Users/qqq/Desktop/SN2N-V0.3.0/examples/denoising2D/data/raw_data'
    model_path = 'C:/Users/qqq/Desktop/SN2N-V0.3.0/examples/denoising2D/data/models'
    infer_mode = '1'
    
    Predict2D_args = [
        '--img_path', img_path,
        '--model_path', model_path,
        '--infer_mode', infer_mode
    ]
    
    
    if len(sys.argv) > 1:
        args = Predict2D()
    else:
        args = Predict2D(Predict2D_args)
    
    print("Parsed arguments:", args)     
    
    ##Step 2: Execute predicting.
    p = Predictor2D(img_path = args.img_path, model_path = args.model_path, infer_mode = args.infer_mode)
    p.execute()
