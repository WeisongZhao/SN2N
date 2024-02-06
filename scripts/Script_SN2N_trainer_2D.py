# -*- coding: utf-8 -*-

import sys
from SN2N.trainer import net2D
from SN2N.get_options import trainer2D
    

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
        
    img_path = 'C:/Users/qqq/Desktop/SN2N-V0.3.0/examples/denoising2D/data/raw_data'
    sn2n_loss = '1'
    bs = '32'
    lr = '2e-4'
    epochs = '100'
    
    trainer2D_args = [
        '--img_path', img_path,
        '--sn2n_loss', sn2n_loss,
        '--bs', bs,
        '--lr', lr,
        '--epochs', epochs
    ]
    
    if len(sys.argv) > 1:
        args = trainer2D()
    else:
        args = trainer2D(trainer2D_args)
    
    print("Parsed arguments:", args) 
    
    
    ##Step 2: Execute training.
    sn2nunet = net2D(img_path = args.img_path, sn2n_loss = args.sn2n_loss, bs = args.bs, lr = args.lr, epochs = args.epochs)
    sn2nunet.train()
