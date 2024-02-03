# -*- coding: utf-8 -*-

import os
import sys
current_script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_script_path, '..', 'SN2N'))
from inference import Predictor2D
from get_options import Predict2D
    

if __name__ == '__main__':
    ##Step 1: Define custom parameters.
    """
    -----Parameters------
    img_path:
        Path of raw images to inference
    """
    
    img_path = 'C:/Users/qqq/Desktop/SN2N-V0.28/examples/simu/data'
    
    Predict2D_args = [
        '--img_path', img_path
    ]
    
    ##Step 2: Execute predicting.
    args = Predict2D(Predict2D_args)
    p = Predictor2D(img_path = args.img_path)
    p.execute()
