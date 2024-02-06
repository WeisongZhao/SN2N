# -*- coding: utf-8 -*-

from SN2N.inference import Predictor3D
from SN2N.get_options import Predict3D
    

if __name__ == '__main__':
    ##Step 1: Define custom parameters.
    """
    -----Parameters------
    img_path:
        Path of raw images to inference
        
    ======Other parameters do not require modification; for details, refer to SN2N.get_options.========
    """
    
    img_path = 'C:/Users/qqq/Desktop/SN2N-V0.29/examples/denoising3D/data/raw_data_per'
    
    Predict3D_args = [
        '--img_path', img_path
    ]
    
    ##Step 2: Execute predicting.
    args = Predict3D(Predict3D_args)
    p = Predictor3D(img_path = args.img_path)
    p.execute()
