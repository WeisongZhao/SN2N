# -*- coding: utf-8 -*-

import os
import torch
import tifffile
import argparse
import numpy as np


def TOTENSOR_(img_datas_np):
    if len(img_datas_np.shape) == 2:
       img_datas_np = np.expand_dims(img_datas_np, axis = 0)
       img_datas_np = np.expand_dims(img_datas_np, axis = 0)
    return torch.from_numpy(img_datas_np).type(torch.FloatTensor)


def normalize_(input):
    img = np.array(input).astype('float32')
    img = img - np.min(img)
    img = img / np.max(img)
    return img

def normalize_tanh(input):
    img = np.array(input).astype('float32')
    img = img * 2 + 1
    img = img - np.min(img)
    img = img / np.max(img)
    return img


def predict_(model, img_path, save_path, epoch, ifGPU=True):
    if ifGPU:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    with torch.no_grad():
        for (root, dirs, files) in os.walk(img_path):
            for j, Ufile in enumerate(files):
                imgpath = os.path.join(root, Ufile)
                image_data = tifffile.imread(imgpath)
                try:
                    [t, x, y] = image_data.shape
                    test_pred_np = np.zeros((t,x,y))
                    for taxial in range(t):
                        datatensor = TOTENSOR_(normalize_(image_data[taxial,:,:]))
                        test_pred = model(datatensor.to(device))
                        test_pred = test_pred.to(torch.device("cpu"))
                        test_pred_np[taxial,:,:] = 255 * normalize_tanh(test_pred.numpy())
                        savepath_final = save_path
                        # savepath_final = save_path + ('%s' %(Ufile))
                        os.makedirs(savepath_final, exist_ok=True)
                    tifffile.imwrite('%s%s_predict_epoch_%d.tif'%(savepath_final,Ufile, epoch),test_pred_np.astype('uint8'))
                    print('Frame: %d'%(j + 1))
                except ValueError:
                    datatensor = TOTENSOR_(normalize_(image_data))
                    test_pred = model(datatensor.to(device))
                    test_pred = test_pred.to(torch.device("cpu"))
                    test_pred_np = 255 * normalize_tanh(test_pred.numpy())
                    savepath_final = save_path
                    # savepath_final = save_path + ('%s' %(Ufile))
                    os.makedirs(savepath_final, exist_ok=True)
                    tifffile.imwrite('%s%s_predict_epoch_%d.tif'%(savepath_final, Ufile, epoch),test_pred_np.astype('uint8'))
                    print('Frame: %d'%(j + 1))
        return 

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_path = args.img_path
    save_path = args.save_path
    model_path = args.model_path

    model = torch.load(model_path, map_location=device)
    predict_(model, img_path, save_path, args.epoch, args.ifGPU)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image Prediction Script")
    parser.add_argument('--cuda_device', type=str, default="0", help='CUDA device ID')
    parser.add_argument('--img_path', type=str, required=True, help='Path to the input images')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the predictions')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--epoch', type=int, default=40, help='Epoch number of the model')
    parser.add_argument('--ifGPU', type=bool, default=True, help='Flag to use GPU or not')

    args = parser.parse_args()
    main(args)
