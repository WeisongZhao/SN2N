# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
from SelfN2N.SN2N_datagen_3D import data_generator

def main_datagen(args):
    train_data_path = os.path.join('../Dataset_3D/', args.train_data_path)
    name = args.name
    save_path = os.path.join('../DL_dataset_3D/SN2N/', args.save_path, 'train/')
    os.makedirs(save_path, exist_ok=True)

    d = data_generator(img_path=train_data_path, save_path=save_path,
                       augment_mode=args.augment_mode, pre_augment_mode=args.pre_augment_mode,
                       img_res=args.img_res, ifx2=args.ifx2, inter_method=args.inter_method)

    d.savedata4folder_agument_multiframe_stack(multi_frame=args.multi_frame, interval=args.interval,
                                               times=args.times, roll=args.roll,
                                               threshold_mode=args.threshold_mode, threshold=args.threshold)
    print('OK')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="3D Data Generator Script")
    parser.add_argument('--train_data_path', type=str, default='noisy/', help='Path to the training data')
    parser.add_argument('--name', type=str, default='noisy_3D', help='Name for the training dataset')
    parser.add_argument('--save_path', type=str, default='noisy_3D/', help='Base path for saving the generated data')
    parser.add_argument('--augment_mode', type=int, default=0, help='Augmentation mode')
    parser.add_argument('--pre_augment_mode', type=int, default=0, help='Pre-augmentation mode in P2P')
    parser.add_argument('--img_res', type=tuple, default=(128, 128), help='Image resolution')
    parser.add_argument('--ifx2', type=bool, default=True, help='Boolean flag for ifx2')
    parser.add_argument('--inter_method', type=str, default='Fourier', help='Interpolation method')
    parser.add_argument('--multi_frame', type=int, default=16, help='Number of frames to stack in multi-frame')
    parser.add_argument('--interval', type=int, default=64, help='Interval for savedata4folder_agument')
    parser.add_argument('--times', type=int, default=0, help='Repeat times of ROI interchange in P2P')
    parser.add_argument('--roll', type=int, default=0, help='Repeat times of interchange for one image in P2P')
    parser.add_argument('--threshold_mode', type=int, default=2, help='Threshold mode to exclude some black patches')
    parser.add_argument('--threshold', type=int, default=0, help='Threshold value to exclude some black patches')

    args = parser.parse_args()
    main_datagen(args)

