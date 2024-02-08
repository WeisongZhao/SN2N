# -*- coding: utf-8 -*-

import argparse

def datagen2D(arguments=None):
    parser = argparse.ArgumentParser(description="SN2N Data Generator Configuration")
    ##============Important=====================
    parser.add_argument('--img_path', 
                        type=str, 
                        required=True, 
                        help='Path to the raw images for training.')
    parser.add_argument('--P2Pmode', 
                        type=int, 
                        default=0, 
                        choices=[0, 1, 2, 3], 
                        help="""Patch2Patch augmentation mode. 
                        0: None; 
                        1: Direct interchange in time; 
                        2: Interchange in a single frame; 
                        3: Interchange in multiple frames but in different regions.""")
    parser.add_argument('--P2Pup', 
                        type=int, 
                        default=1, 
                        help='Increase the dataset to its (1 + P2Pup) times size.')
    parser.add_argument('--BAmode', 
                        type=int, 
                        default=0, 
                        choices=[0, 1, 2], 
                        help="""Basic augmentation mode. 
                        0: None; 
                        1: Double the dataset with random rotate and flip; 
                        2: Eightfold the dataset with random rotate and flip.""")
    parser.add_argument('--SWsize', 
                        type=int, 
                        default=64, 
                        help='Interval in pixels of the sliding window for generating image patches.')
    
    ##===========No need to change=============
    parser.add_argument('--SWmode', 
                        type=int, 
                        default=1, 
                        choices=[0, 1], 
                        help="""Threshold mode to exclude some black patches. 
                        0: Set the actual threshold directly; 
                        1: The actual threshold is set to the sum of the image's average value and the `SWfilter`.""")
    parser.add_argument('--SWfilter', 
                        type=int, 
                        default=1, 
                        help="""Threshold for excluding patches with black areas.
                        The actual threshold value is determined according to the description provided for `SWmode`.""")
    parser.add_argument('--P2Ppatch', 
                        type=str, 
                        default='64', 
                        help='ROI size for interchange, specified as "width,height".')
    parser.add_argument('--img_patch', 
                        type=str, 
                        default='128', 
                        help='Image patch size, specified as "width,height".')
    parser.add_argument('--ifx2', 
                        type=bool,
                        default=True, 
                        help='Flag to indicate rescaling to the original size. If set, rescales; otherwise, does not rescale.')
    parser.add_argument('--inter_mode', 
                        type=str, 
                        default='Fourier', 
                        choices=['Fourier', 'bilinear'], 
                        help='Scaling method. "Fourier" for Fourier rescaling, "bilinear" for spatial rescaling.')
    
    
    args, _ = parser.parse_known_args(arguments) 

    return args


def datagen3D(arguments=None):
    parser = argparse.ArgumentParser(description="SN2N Data Generator Configuration")
    ##============Important=====================
    parser.add_argument('--img_path', 
                        type=str, 
                        required=True, 
                        help='Path to the raw images for training.')
    parser.add_argument('--P2Pmode', 
                        type=int, 
                        default=0, 
                        choices=[0, 1, 2, 3], 
                        help="""Patch2Patch augmentation mode. 
                        0: None; 
                        1: Direct interchange in time; 
                        2: Interchange in a single frame; 
                        3: Interchange in multiple frames but in different regions.""")
    parser.add_argument('--P2Pup', 
                        type=int, 
                        default=1, 
                        help='Increase the dataset to its (1 + P2Pup) times size.')
    parser.add_argument('--BAmode', 
                        type=int, 
                        default=0, 
                        choices=[0, 1, 2], 
                        help="""Basic augmentation mode. 
                        0: None; 
                        1: Double the dataset with random rotate and flip; 
                        2: Eightfold the dataset with random rotate and flip.""")
    parser.add_argument('--SWsize', 
                        type=int, 
                        default=64, 
                        help='Interval in pixels of the sliding window for generating image patches.')
    
    ##===========No need to change=============
    parser.add_argument('--SWmode', 
                        type=int, 
                        default=1, 
                        choices=[0, 1], 
                        help="""Threshold mode to exclude some black patches. 
                        0: Set the actual threshold directly; 
                        1: The actual threshold is set to the sum of the image's average value and the `SWfilter`.""")
    parser.add_argument('--SWfilter', 
                        type=int, 
                        default=1, 
                        help="""Threshold for excluding patches with black areas.
                        The actual threshold value is determined according to the description provided for `SWmode`.""")
    parser.add_argument('--P2Ppatch', 
                        type=str, 
                        default='64', 
                        help='ROI size for interchange, specified as "width,height".')
    parser.add_argument('--vol_patch', 
                        type=str, 
                        default='16,128,128', 
                        help='Image patch size, specified as "width,height".')
    parser.add_argument('--ifx2', 
                        type=bool,
                        default=True, 
                        help='Flag to indicate rescaling to the original size. If set, rescales; otherwise, does not rescale.')
    parser.add_argument('--inter_mode', 
                        type=str, 
                        default='Fourier', 
                        choices=['Fourier', 'bilinear'], 
                        help='Scaling method. "Fourier" for Fourier rescaling, "bilinear" for spatial rescaling.')
    
    args, _ = parser.parse_known_args(arguments) 

    return args

def trainer2D(arguments=None):
    parser = argparse.ArgumentParser(description="SelfN2N Neural Network Configuration")
    ##============Important=====================
    parser.add_argument('--img_path', 
                        type=str, 
                        required=True, 
                        help='Path to the raw images for training.')
    parser.add_argument('--sn2n_loss', 
                        type=float, 
                        default=1, 
                        help='Weight of the self-constrained loss.')
    parser.add_argument('--bs', 
                        type=int, 
                        default=32, 
                        help='Training batch size.')
    parser.add_argument('--lr', 
                        type=float, 
                        default=2e-4, 
                        help='Learning rate.')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=100, 
                        help='Total number of training epochs.')
    
    ##===========No need to change=============
    parser.add_argument('--img_patch', 
                        type=str, 
                        default='128', 
                        help='Patch size for training, specified as "width,height".')
    parser.add_argument('--if_alr',
                        type=bool,
                        default=True,
                        help='Flag to use adaptive learning rate. If set, adaptive learning rate will be used; otherwise, it will not.')

    args, _ = parser.parse_known_args(arguments) 

    return args


def trainer3D(arguments=None):
    parser = argparse.ArgumentParser(description="SelfN2N Neural Network Configuration")
    ##============Important=====================
    parser.add_argument('--img_path', 
                        type=str, 
                        required=True, 
                        help='Path to the raw images for training.')
    parser.add_argument('--sn2n_loss', 
                        type=float, 
                        default=1, 
                        help='Weight of the self-constrained loss.')
    parser.add_argument('--bs', 
                        type=int, 
                        default=4, 
                        help='Training batch size.')
    parser.add_argument('--lr', 
                        type=float, 
                        default=2e-4, 
                        help='Learning rate.')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=100, 
                        help='Total number of training epochs.')
    
    ##===========No need to change=============
    parser.add_argument('--vol_patch', 
                        type=str, 
                        default='16,128,128', 
                        help='Patch size for training, specified as "width,height".')
    parser.add_argument('--if_alr', 
                        type=bool,
                        default=True,
                        help='Flag to use adaptive learning rate. If set, adaptive learning rate will be used; otherwise, it will not.')
    
    args, _ = parser.parse_known_args(arguments) 

    return args


def Predict2D(arguments=None):
    parser = argparse.ArgumentParser(description="Image Prediction Script")
    ##============Important=====================
    parser.add_argument('--img_path', 
                        type=str, 
                        required=True, 
                        help='Path to the raw images for training.')
    parser.add_argument('--model_path', 
                        type=str, 
                        required=True, 
                        help='Path of model for inference.')
    parser.add_argument('--infer_mode', 
                        type=int, 
                        default=0, 
                        choices=[0, 1], 
                        help="""Prediction Mode. 
                        0: Predict the results of all models generated during training 
                        under the default "models" directory on the img_path.                
                        1: Predict the results of the models provided by the user under 
                        the given model_path on the Img_path provided by the user.""")
    

    args, _ = parser.parse_known_args(arguments) 

    return args

def Predict3D(arguments=None):
    parser = argparse.ArgumentParser(description="Image Prediction Script")
    ##============Important=====================
    parser.add_argument('--img_path', 
                        type=str, 
                        required=True, 
                        help='Path to the raw images for training.')
    parser.add_argument('--model_path', 
                        type=str, 
                        required=True, 
                        help='Path of model for inference.')
    parser.add_argument('--infer_mode', 
                        type=int, 
                        default=0, 
                        choices=[0, 1], 
                        help="""Prediction Mode. 
                        0: Predict the results of all models generated during training 
                        under the default "models" directory on the img_path.                
                        1: Predict the results of the models provided by the user under 
                        the given model_path on the Img_path provided by the user.""")
    parser.add_argument('--overlap_shape', 
                        type=str, 
                        default = '2,256,256', 
                        help='Overlap shape in 3D stitching prediction.')
    
    args, _ = parser.parse_known_args(arguments)  

    return args



def execute2D(arguments=None):
    parser = argparse.ArgumentParser(description="SelfN2N Data Generator Configuration")
    ##============Important=====================
    parser.add_argument('--img_path', 
                        type=str, 
                        required=True, 
                        help='Path to the raw images for training.')
    parser.add_argument('--P2Pmode', 
                        type=int, 
                        default=0, 
                        choices=[0, 1, 2, 3], 
                        help="""Patch2Patch augmentation mode. 
                        0: None; 
                        1: Direct interchange in time; 
                        2: Interchange in a single frame; 
                        3: Interchange in multiple frames but in different regions.""")
    parser.add_argument('--P2Pup', 
                        type=int, 
                        default=1, 
                        help='Increase the dataset to its (1 + P2Pup) times size.')
    parser.add_argument('--BAmode', 
                        type=int, 
                        default=0, 
                        choices=[0, 1, 2], 
                        help="""Basic augmentation mode. 
                        0: None; 
                        1: Double the dataset with random rotate and flip; 
                        2: Eightfold the dataset with random rotate and flip.""")
    parser.add_argument('--SWsize', 
                        type=int, 
                        default=64, 
                        help='Interval in pixels of the sliding window for generating image patches.')
    parser.add_argument('--sn2n_loss', 
                        type=float, 
                        default=1, 
                        help='Weight of the self-constrained loss.')
    parser.add_argument('--bs', 
                        type=int, 
                        default=32, 
                        help='Training batch size.')
    parser.add_argument('--lr', 
                        type=float, 
                        default=2e-4, 
                        help='Learning rate.')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=100, 
                        help='Total number of training epochs.')
    parser.add_argument('--model_path', 
                        type=str, 
                        required=True, 
                        help='Path of model for inference.')
    parser.add_argument('--infer_mode', 
                        type=int, 
                        default=0, 
                        choices=[0, 1], 
                        help="""Prediction Mode. 
                        0: Predict the results of all models generated during training 
                        under the default "models" directory on the img_path.                
                        1: Predict the results of the models provided by the user under 
                        the given model_path on the Img_path provided by the user.""")
    
    ##===========No need to change=============
    parser.add_argument('--SWmode', 
                        type=int, 
                        default=1, 
                        choices=[0, 1], 
                        help="""Threshold mode to exclude some black patches. 
                        0: Set the actual threshold directly; 
                        1: The actual threshold is set to the sum of the image's average value and the `SWfilter`.""")
    parser.add_argument('--SWfilter', 
                        type=int, 
                        default=1, 
                        help="""Threshold for excluding patches with black areas.
                        The actual threshold value is determined according to the description provided for `SWmode`.""")
    parser.add_argument('--P2Ppatch', 
                        type=str, 
                        default='64', 
                        help='ROI size for interchange, specified as "width,height".')
    parser.add_argument('--img_patch', 
                        type=str, 
                        default='128', 
                        help='Image patch size, specified as "width,height".')
    parser.add_argument('--ifx2', 
                        type=bool,
                        default=True, 
                        help='Flag to indicate rescaling to the original size. If set, rescales; otherwise, does not rescale.')
    parser.add_argument('--inter_mode', 
                        type=str, 
                        default='Fourier', 
                        choices=['Fourier', 'bilinear'], 
                        help='Scaling method. "Fourier" for Fourier rescaling, "bilinear" for spatial rescaling.')
    parser.add_argument('--if_alr',
                        type=bool,
                        default=True,
                        help='Flag to use adaptive learning rate. If set, adaptive learning rate will be used; otherwise, it will not.')

    
    args, _ = parser.parse_known_args(arguments)  

    return args


def execute3D(arguments=None):
    parser = argparse.ArgumentParser(description="SelfN2N Data Generator Configuration")
    ##============Important=====================
    parser.add_argument('--img_path', 
                        type=str, 
                        required=True, 
                        help='Path to the raw images for training.')
    parser.add_argument('--P2Pmode', 
                        type=int, 
                        default=0, 
                        choices=[0, 1, 2, 3], 
                        help="""Patch2Patch augmentation mode. 
                        0: None; 
                        1: Direct interchange in time; 
                        2: Interchange in a single frame; 
                        3: Interchange in multiple frames but in different regions.""")
    parser.add_argument('--P2Pup', 
                        type=int, 
                        default=1, 
                        help='Increase the dataset to its (1 + P2Pup) times size.')
    parser.add_argument('--BAmode', 
                        type=int, 
                        default=0, 
                        choices=[0, 1, 2], 
                        help="""Basic augmentation mode. 
                        0: None; 
                        1: Double the dataset with random rotate and flip; 
                        2: Eightfold the dataset with random rotate and flip.""")
    parser.add_argument('--SWsize', 
                        type=int, 
                        default=64, 
                        help='Interval in pixels of the sliding window for generating image patches.')
    parser.add_argument('--sn2n_loss', 
                        type=float, 
                        default=1, 
                        help='Weight of the self-constrained loss.')
    parser.add_argument('--bs', 
                        type=int, 
                        default=32, 
                        help='Training batch size.')
    parser.add_argument('--lr', 
                        type=float, 
                        default=2e-4, 
                        help='Learning rate.')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=100, 
                        help='Total number of training epochs.')
    parser.add_argument('--model_path', 
                        type=str, 
                        required=True, 
                        help='Path of model for inference.')
    parser.add_argument('--infer_mode', 
                        type=int, 
                        default=0, 
                        choices=[0, 1], 
                        help="""Prediction Mode. 
                        0: Predict the results of all models generated during training 
                        under the default "models" directory on the img_path.                
                        1: Predict the results of the models provided by the user under 
                        the given model_path on the Img_path provided by the user.""")
    parser.add_argument('--overlap_shape', 
                        type=str, 
                        default = '2,256,256', 
                        help='Overlap shape in 3D stitching prediction.')
                        
    ##===========No need to change=============
    parser.add_argument('--SWmode', 
                        type=int, 
                        default=1, 
                        choices=[0, 1], 
                        help="""Threshold mode to exclude some black patches. 
                        0: Set the actual threshold directly; 
                        1: The actual threshold is set to the sum of the image's average value and the `SWfilter`.""")
    parser.add_argument('--SWfilter', 
                        type=int, 
                        default=1, 
                        help="""Threshold for excluding patches with black areas.
                        The actual threshold value is determined according to the description provided for `SWmode`.""")
    parser.add_argument('--P2Ppatch', 
                        type=str, 
                        default='64', 
                        help='ROI size for interchange, specified as "width,height".')
    parser.add_argument('--vol_patch', 
                        type=str, 
                        default='16,128,128', 
                        help='Image patch size, specified as "width,height".')
    parser.add_argument('--ifx2', 
                        type=bool,
                        default=True, 
                        help='Flag to indicate rescaling to the original size. If set, rescales; otherwise, does not rescale.')
    parser.add_argument('--inter_mode', 
                        type=str, 
                        default='Fourier', 
                        choices=['Fourier', 'bilinear'], 
                        help='Scaling method. "Fourier" for Fourier rescaling, "bilinear" for spatial rescaling.')
    parser.add_argument('--if_alr', 
                        type=bool,
                        default=True,
                        help='Flag to use adaptive learning rate. If set, adaptive learning rate will be used; otherwise, it will not.')
    
    args, _ = parser.parse_known_args(arguments)  

    return args

