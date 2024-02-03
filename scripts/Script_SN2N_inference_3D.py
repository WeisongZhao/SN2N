# -*- coding: utf-8 -*-

import os
import sys
current_script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_script_path, '..', 'SN2N'))
from inference import Predictor3D
from get_options import Predict3D
    

if __name__ == '__main__':
    ##Step 1: Define custom parameters.
    """
    -----Parameters------
    img_path:
        Path of raw images to inference
        
    ======Other parameters do not require modification; for details, refer to SN2N.get_options.========
    """
    
    img_path = 'C:/Users/qqq/Desktop/SN2N-V0.28/examples/xyzt/data'
    
    Predict3D_args = [
        '--img_path', img_path
    ]
    
    ##Step 2: Execute predicting.
    args = Predict3D(Predict3D_args)
    p = Predictor3D(img_path = args.img_path)
    p.execute()
