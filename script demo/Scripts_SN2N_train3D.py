# -*- coding: utf-8 -*-

import os
import sys
import torch
import argparse
from SelfN2N.selfn2n_3D import SelfN2N

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
    torch.cuda.empty_cache()

    dataset_name = args.dataset_name
    epochs = args.epochs
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    reg = args.reg
    reg_sparse = args.reg_sparse
    prefix = args.prefix

    tests_name = f"{prefix}{dataset_name}_EPOCH{epochs}_BS{train_batch_size}_LOSSconsis_{reg:.2f}_LOSSsparse{reg_sparse:.2f}_"

    os.makedirs(os.path.join('../images_3D/', tests_name), exist_ok=True)
    os.makedirs(os.path.join('../images_3D/', tests_name, 'weights'), exist_ok=True)
    os.makedirs(os.path.join('../images_3D/', tests_name, 'images'), exist_ok=True)

    sn2nunet = SelfN2N(dataset_name=dataset_name, tests_name=tests_name, reg=reg, reg_sparse=reg_sparse, 
                        constrained_type='L1', epochs=epochs, train_batch_size=train_batch_size, 
                        ifadaptive_lr=args.ifadaptive_lr, test_batch_size=test_batch_size, img_res=args.img_res)
    sn2nunet.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SelfN2N Training Script")
    parser.add_argument('--cuda_device', type=str, default="0", help='CUDA device ID')
    parser.add_argument('--dataset_name', type=str, default='noisy_3D', help='Dataset name')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--train_batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='Test batch size')
    parser.add_argument('--reg', type=float, default=1, help='Regularization parameter')
    parser.add_argument('--reg_sparse', type=float, default=0, help='Sparse regularization parameter')
    parser.add_argument('--prefix', type=str, default='0123_', help='Prefix for tests name')
    parser.add_argument('--ifadaptive_lr', type=bool, default=False, help='Flag for adaptive learning rate')
    parser.add_argument('--img_res', type=lambda s: tuple(map(int, s.split(','))), default=(128, 128), help='Image resolution in format H,W (e.g., 128,128)')

    args = parser.parse_args()
    main(args)

