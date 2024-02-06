# -*- coding: utf-8 -*-

from SN2N.datagen import generator2D, generator3D
from SN2N.trainer import net2D, net3D
from SN2N.inference import Predictor2D, Predictor3D

def SN2Nexecute_2D(args):
    d = generator2D(img_path=args.img_path, P2Pmode = args.P2Pmode, P2Pup = args.P2Pup, BAmode = args.BAmode, SWsize = args.SWsize,
                    SWfilter= args.SWfilter, P2Ppatch = args.P2Ppatch, img_patch = args.img_patch, ifx2 = args.ifx2, inter_mode = args.inter_mode)
    d.execute()
    
    sn2nunet = net2D(img_path = args.img_path, sn2n_loss = args.sn2n_loss, bs = args.bs, lr = args.lr, epochs = args.epochs,
                     img_patch = args.img_patch,  if_alr = args.if_alr)
    sn2nunet.train()
    
    p = Predictor2D(img_path = args.img_path)
    p.execute()
    
def SN2Nexecute_3D(args):
    d = generator3D(img_path=args.img_path, P2Pmode = args.P2Pmode, P2Pup = args.P2Pup, BAmode = args.BAmode, SWsize = args.SWsize,
                    SWfilter= args.SWfilter, P2Ppatch = args.P2Ppatch, img_patch = args.img_patch, ifx2 = args.ifx2, inter_mode = args.inter_mode)
    d.execute()
    
    sn2nunet = net3D(img_path = args.img_path, sn2n_loss = args.sn2n_loss, bs = args.bs, lr = args.lr, epochs = args.epochs,
                     img_patch = args.img_patch,  if_alr = args.if_alr)
    sn2nunet.train()
    
    p = Predictor3D(img_path = args.img_path)
    p.execute()
