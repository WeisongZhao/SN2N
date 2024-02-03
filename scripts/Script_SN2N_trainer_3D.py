# -*- coding: utf-8 -*-

import os
import sys
current_script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_script_path, '..', 'SN2N'))
from trainer import net3D
from get_options import trainer3D
    

if __name__ == '__main__':
    ##Step 1: Define custom parameters.
    """
    -----Parameters------
    =====Important==========
    img_path:
        Path of raw images to train.
    sn2n_loss:
        Weight of self-constrained loss.
        {default: 1}
    bs:
        Training batch size.
        {default: 32}
    lr:
        Learning rate
        {default: 2e-4}.
    epochs:
        Total number of training epochs.
        {default: 100}.
        
    ======Other parameters do not require modification; for details, refer to SN2N.get_options.========
    """
        
    img_path = 'C:/Users/qqq/Desktop/SN2N-V0.28/examples/xyzt/data'
    sn2n_loss = '1'
    bs = '32'
    lr = '2e-4'
    epochs = '100'
    
    trainer3D_args = [
        '--img_path', img_path,
        '--sn2n_loss', sn2n_loss,
        '--bs', bs,
        '--lr', lr,
        '--epochs', epochs
    ]
    
    args = trainer3D(trainer3D_args)
    
    ##Step 2: Execute training.
    sn2nunet = net3D(img_path = args.img_path)
    sn2nunet.train()
