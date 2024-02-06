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
    """
    
    img_path = 'C:/Users/qqq/Desktop/SN2N-V0.3.0/examples/denoising2D/data/raw_data'
    
    Predict2D_args = [
        '--img_path', img_path
    ]
    
    
    if len(sys.argv) > 1:
        args = Predict2D()
    else:
        args = Predict2D(Predict2D_args)
    
    print("Parsed arguments:", args)     
    
    ##Step 2: Execute predicting.
    p = Predictor2D(img_path = args.img_path)
    p.execute()
