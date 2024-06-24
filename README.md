[![Github commit](https://img.shields.io/github/last-commit/WeisongZhao/SN2N)](https://github.com/WeisongZhao/SN2N/)
[![Github All Releases](https://img.shields.io/github/downloads/WeisongZhao/SN2N/total.svg)](https://github.com/WeisongZhao/SN2N/releases/tag/v0.3.2/)
[![License](https://img.shields.io/github/license/WeisongZhao/SN2N)](https://github.com/WeisongZhao/SN2N/blob/master/LICENSE/)
[![paper](https://img.shields.io/badge/paper-nat.%20methods-black.svg)](https://www.nature.com/nmeth/)
[![releases](https://img.shields.io/badge/release-v0.3.5-FF6600.svg)](https://github.com/WeisongZhao/SN2N/releases/tag/v0.3.5/)
<br>

[![Twitter](https://img.shields.io/twitter/follow/QuLiying?label=liying)](https://twitter.com/QuLiying)
[![Twitter](https://img.shields.io/twitter/follow/weisong_zhao?label=weisong)](https://twitter.com/weisong_zhao)
[![GitHub stars](https://img.shields.io/github/stars/WeisongZhao/SN2N?style=social)](https://github.com/WeisongZhao/SN2N/) 



<p>
<h1 align="center"><font color="#FF6600">S</font>N2N</h1>
<h5 align="center">Self-inspired learning to denoise for live-cell super-resolution microscopy.</h5>
<h6 align="right">v0.3.5</h6>
</p>





<br>


<p>
<img src='./imgs/4color-small.gif' align="left" width=190>
</p>
<br>


This repository is for our developed self-inspired Noise2Noise (SN2N) learning-to-denoise engine, and it will be in continued development. It is distributed as accompanying software for publication: [Liying Qu et al. Self-inspired learning to denoise for live-cell super-resolution microscopy, bioRxiv (2024)](https://doi.org/10.1101/2024.01.23.576521). Please cite SN2N in your publications, if it helps your research.

<br><br><br>

<div align="center">

✨ [**Introduction**](#-Introduction) **|**  🔧 [**Installation**](#-Installation)  **|** 🚀 [**Overall**](#-Overall)**|** 🎨 [**Dataset**](#-Dataset) **|**  💻 [**Training**](#-Training) **|** ⚡ [**Inference**](#-Inference) **|** 🚩 [**Execution**](#-Execution) **|** 🏆 [**Models**](#-Models) **|**&#x1F308; [**Resources**](#-Resources)

</div>

---

## Introduction

<p>
<img src='./imgs/SN2N-workflow.png' align="right" width=500>
</p>

Our SN2N is fully competitive with the supervised learning methods and overcomes the need for large dataset and clean ground-truth. **First**, we create a self-supervised data generation strategy based on super-resolution images' spatial redundancy, using a diagonal resampling step followed by a Fourier interpolation for single-frame Noise2Noise. **Second**, we have taken a step further by ushering in a self-constrained learning process to enhance the performance and data-efficiency. **Finally**, we provide a Patch2Patch data augmentation (random patch transformations in multiple dimensions) to further improve the data efficiency.


## 🔧 Installation

### Tested platform
  - Python = 3.7.6, Pytorch = 1.12.0 (`Win 10`, `128 GB RAM`, `NVIDIA RTX 4090 24 GB`, `CUDA 11.6`)

### Dependencies 
  - Python >= 3.6
  - PyTorch >= 1.10
    
### Instruction

1. Clone the repository.

    ```bash
    git clone https://github.com/WeisongZhao/SN2N.git
    cd SN2N    
    ```

2. Create a virtual environment and install PyTorch and other dependencies. Please select the correct Pytorch version that matches your CUDA version from [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/). 

    Users can set up the environment directly by installing the packages listed in the (**requirements.txt**) file. The packages required by the environment have also been uploaded to the requirements.

    ```bash
    $ conda create -n SN2N python=3.7.6
    $ conda activate SN2N
    
    $ pip install -r requirements.txt
    ```
## 🚀 Overall

We have **two execution modes** for SN2N: The first one is step by step, which involves (**dataset generation, training, and then inference.**)  The second one is directly invoking (**SN2Nexecute.**)

We have provided **two examples** of denoising in 2D and 3D, along with the datasets used for training and validation of SN2N. The 2D dataset consists of synthenic microtubules, represented by 2048 x 2048 pixels with a 32.5 nm pixel size. **The 2D dataset can be found at 'examples/denoising2D/data' path.** The 3D dataset consists of outer mitochondrial membrane (OMM) network imaging of live COS-7 cells labeled with Tom20-mCherry on SD-SIM sysytem. **The 3D dataset** has a size of 101 * 1478 * 1137 pixels, with a pixel size of 38.23 nm, and  **is avaliable on Google drive at** https://drive.google.com/drive/folders/1TI69_SkWC8Ghs6p-6uW9kKI43oMVwV-F


## 🎨 Dataset
### 0. Percentile normalization for data under ultralow SNR (optional)

For the ultralow SNR data with ultrahigh baseline signal and a number of hot pixels, we adapted the routinely used percentile normalization before the data generation step to remove the smooth background or hot pixels. 

There are **two ways to execute percentile normalization**.  One option is to utilize the **percentile normalization Fiji plugin, available for easy access:** https://github.com/WeisongZhao/percentile_normalization.imagej. Users can directly remove the ultra-strong baseline signal before training using this ready-to-use plugin. Alternatively, users can employ the following **Python scripts** for percentile normalization.

```
---------import package--------
import tifffile
import numpy as np
from SN2N.utils import *

---------parameter------------
## pmin and pmax represent mapping the data range to a specified percentage range.
## In the most applications, the pmin and pmax were assigned as 0% and 99.999%.
## For ultralow SNR data, we set pmin and pmax as 20% and 99.9%, respectively.

pmin = 30
pmax = 99.8
imgpath = 'raw.tif'
save_path = 'raw_per.tif'

image_data = tifffile.imread(imgpath)

## 3D percentile normalization
try:
    [t, x, y] = image_data.shape
    for tt in range(t):
        image_data_single = image_data[tt, :, :] 
        image_data_single = np.squeeze(image_data_single)
        image_data_single1 = normalize_percentage(x = image_data_single, pmin = pmin, pmax = pmax, axis=None, clip=True, eps=1e-20, dtype=np.float32)
        image_data_single2 = 255*image_data_single1
        tifffile.imsave(save_path, image_data_single2.astype('uint8'))
        
## 2D percentile normalization        
except ValueError:
        image_data = tifffile.imread(imgpath)
        image_data = normalize_percentage(x = image_data, pmin = pmin, pmax = pmax, axis=None, clip=True, eps=1e-20, dtype=np.float32)
        image_data = 255*image_data
        tifffile.imsave(save_path, image_data)        
    
print("Data preprocessing is completed.")
```


### 1. Self-supervised data generation
Our SN2N is adaptable to both 2D (**xy**) and 3D (**xyz**) datasets. You can use your own data or our uploaded to generate 2D / 3D SN2N data pairs. 

Users can run the script after customizing parameters in Script_SN2N_datagen_2D.py (Script_SN2N_datagen_3D.py) or run directly from the command line.

#### 2D data generation
```
python Script_SN2N_datagen_2D.py --img_path "Users own path/data/raw_data" --P2Pmode "1" --P2Pup "1" --BAmode "1" --SWsize "64"        
```

#### 3D data generation
```
python Script_SN2N_datagen_3D.py --img_path "Users own path/data/raw_data" --P2Pmode "1" --P2Pup "1" --BAmode "1" --SWsize "64"    
```

#### Parameters instructions

The key parameters for 2D data generation and 3D data generation are consistent. There are also other parameters that do not require user modification. Detailed explanations can be found in the SN2N.get_options.datagen2D / SN2N.get_options.datagen3D function. 

```
    -----Parameters------
    =====Important==========
    img_path:
        Path of raw images to train.
    P2Pmode(0 ~ 3):
        Augmentation mode for Patch2Patch.
        0: NONE; 
        1: Direct interchange in t;
        2: Interchange in single frame;
        3: Interchange in multiple frame but in different regions;
        {default: 0}
    P2Pup:
        Increase the dataset to its (1 + P2Pup) times size.
        {default: 0}
    BAmode(0 ~ 2):
        Basic augmentation mode.
        0: NONE; 
        1: double the dataset with random rotate&flip;
        2: eightfold the dataset with random rotate&flip;
        {default: 0} 
    SWsize:
        Interval pixel of sliding window for generating image pathes.
        {default: 64}
        
    ======Other parameters do not require modification; ======
    ======for details, refer to SN2N.get_options.========
    
```

## 💻 Training 

### 1. Prepare the data  

You can use SN2N data pairs for SN2N learning

### 2. Start training

Users can run the script after customizing parameters in Script_SN2N_trainer_2D.py (Script_SN2N_trainer_3D.py) or run directly from the command line.

#### 2D data training

```
python Script_SN2N_trainer_2D.py --img_path "Users own path/data/raw_data" --sn2n_loss "1" --bs "32" --lr "2e-4" --epochs "100"
```

#### 3D data training

```
python Script_SN2N_trainer_3D.py --img_path "Users own path/data/raw_data" --sn2n_loss "1" --bs "4" --lr "2e-4" --epochs "100" 
```

#### Parameters instructions

The key parameters for 2D trainer and 3D trainer are consistent. There are also other parameters that do not require user modification. Detailed explanations can be found in the SN2N.get_options.trainer2D/ SN2N.get_options.trainer3D function.

```
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
        
    ======Other parameters do not require modification; ======
    ======for details, refer to SN2N.get_options.========
    
```

## ⚡ Inference
### 1. Prepare models

Before inference, you should have trained your own model

### 2. Test models

Users can run the script after customizing parameters in Script_SN2N_inference_2D.py (Script_SN2N_inference_3D.py) or run directly from the command line.

#### 2D inference

  ```
python Script_SN2N_inference_2D.py --img_path "Users own path/data/raw_data" --model_path "Users own path/data/model" --infer_mode "0"
  ```

#### 3D inference
In 3D prediction tasks, we use the method of stitching predictions to avoid issues of memory overflow

  ```
  python Script_SN2N_inference_3D.py --img_path "Users own path/data/raw_data" --model_path "Users own path/data/model" --infer_mode "0" --overlap_shape "2,256,256"
  ```

#### Parameters instructions

The key parameters for 2D inference and 3D inference are nearly consistent except for'overlap_shape'. There are also other parameters that do not require user modification. Detailed explanations can be found in the SN2N.get_options.Predict2D/ SN2N.get_options.Predict3D function.

```
    -----2D/3D inference Parameters------
    img_path:
        Path of raw images to inference
    model_path:
        Path of model for inference
    infer_mode:
        Prediction Mode
        0: Predict the results of all models generated during training 
        under the default "models" directory on the img_path.                
        1: Predict the results of the models provided by the user under 
        the given model_path on the Img_path provided by the user.
        
    -----3D execution unique Parameters------
    overlap_shape:
        Overlap shape in 3D stitching prediction.
        {default: '2, 256, 256'}    
```

## 🚩 Execution

### 1. Self-supervised data generation
Our SN2N denoiser has been integrated into the SN2Nexecute.py function, allowing users to input their **allowing users to input their own parameters for denoising with just one click**.

Users can run the script after **customizing parameters** in Script_SN2Nexecute_2D.py (Script_SN2Nexecute_3D.py).
#### 2D execution
```
python Script_SN2Nexecute_2D.py --img_path "Users own path/data/raw_data" --P2Pmode "1" --P2Pup "1" --BAmode "1" --SWsize "64" --sn2n_loss "1" --bs "32" --lr "2e-4" --epochs "100" --model_path "Users own path/data/model" --infer_mode "0"
```

#### 3D execution
```
python Script_SN2Nexecute_3D.py --img_path "Users own path/data/raw_data" --P2Pmode "1" --P2Pup "1" --BAmode "1" --SWsize "64" --sn2n_loss "1" --bs "32" --lr "2e-4" --epochs "100" --model_path "Users own path/data/model" --infer_mode "0" --overlap_shape "2,256,256"
```
#### Parameters instructions

The key parameters for 2D  execute and 3D execution are nearly consistent execept for 'overlap_shape'. There are also other parameters that do not require user modification. Detailed explanations can be found in the SN2N.get_options.execute2D / SN2N.get_options.execute 3D function. 

```
    -----2D/3D execution Parameters------
    =====Important==========
    img_path:
        Path of raw images to train.
    P2Pmode(0 ~ 3):
        Augmentation mode for Patch2Patch.
        0: NONE; 
        1: Direct interchange in t;
        2: Interchange in single frame;
        3: Interchange in multiple frame but in different regions;
        {default: 0}
    P2Pup:
        Increase the dataset to its (1 + P2Pup) times size.
        {default: 0}
    BAmode(0 ~ 2):
        Basic augmentation mode.
        0: NONE; 
        1: double the dataset with random rotate&flip;
        2: eightfold the dataset with random rotate&flip;
        {default: 0} 
    SWsize:
        Interval pixel of sliding window for generating image pathes.
        {default: 64}
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
    model_path:
        Path of model for inference
    infer_mode:
        Prediction Mode
        0: Predict the results of all models generated during training 
        under the default "models" directory on the img_path.                
        1: Predict the results of the models provided by the user under 
        the given model_path on the Img_path provided by the user.
        
    -----3D execution unique Parameters------
    overlap_shape:
        Overlap shape in 3D stitching prediction.
        {default: '2, 256, 256'} 
        
    ======Other parameters do not require modification; ======
    ======for details, refer to SN2N.get_options.========
    
```

## 🏆 Models

We  have provided 12 generalized pre-trained models for specfic tasks and hope this will be helpful to the community.

|            No.            | Data                                                         | Microscopy system | Pixel size | Network  |
| :-----------------------: | ------------------------------------------------------------ | :---------------: | :--------: | :------: |
|        1 (Fig. 2l)        | fixed COS-7 cells of [lysosomes](https://zenodo.org/record/12518397/files/fixed_lysosomes_model_4_11_full.pth?download=1) labeled with LAMP1-EGFP       |      SD-SIM       |  32.5 nm   | 2D U-Net |
|        2 (Fig. 2m)        | fixed COS-7 cells of [mitochondria](https://zenodo.org/record/12518397/files/fixed_mito_model_4_8_full.pth?download=1) labeled with Tom20-mGold1  |      SD-SIM       |  32.5 nm   | 2D U-Net |
|        3 (Fig. 4b)        | Live COS-7 cells of [mitochondria](https://zenodo.org/record/12518397/files/Live_mito_3D_model_5_26_full.pth?download=1) labeled with Tom20–mCherry  |      SD-SIM       |  38.23 nm  | 3D U-Net |
|        4 (Fig. 4e)        | Live COS-7 cells of [mitochondria](https://zenodo.org/record/12518397/files/Live_mito_4D_model_5_31_full.pth?download=1) labeled with Tom20–mCherry  |      SD-SIM       |  38.23 nm  | 3D U-Net |
|        5 (Fig. 4g)        | Live COS-7 cells, [mitochondria](https://zenodo.org/record/12518397/files/Live_mito_5D_model_10_31_full.pth?download=1) labeled with mGold-Mito-N-7, [ER](https://zenodo.org/record/12518397/files/Live_ER_5D_model_11_1_full.pth?download=1) labeled with DsRed-ER, and the [nucleus](https://zenodo.org/record/12518397/files/Live_nuclear_5D_model_11_1_full.pth?download=1) labeled with SPY650-DNA |      SD-SIM       |  38.23 nm  | 3D U-Net |
|        6 (Fig. 5g)        | Live COS-7 cells, [MT](https://zenodo.org/record/12518397/files/STED_live_MT_model_12_30_full.pth?download=1) labeled with SiR-Tubulin, [Actin](https://zenodo.org/record/12518397/files/STED_live_actin_model_5_10_full.pth?download=1) labeled with Lifeact-EGFP, and the [ER](https://zenodo.org/record/12518397/files/STED_live_ER_model_12_30_full.pth?download=1) labeled with Sec61β–EGFP |       STED        |   16 nm    | 2D U-Net |
| 7 (Extended Data Fig. 6a) | Live COS-7 cells, [ER](https://zenodo.org/record/12518397/files/Live_ER_exfig6_model_4_9_full.pth?download=1) labeled with Hoechst and the [mitochondira](https://zenodo.org/record/12518397/files/Live_mito_exfig6_model_4_10_full.pth?download=1) labeled with MitoTracker Deep Red |   EMCCD SD-SIM    |   94 nm    | 3D U-Net |



## Version

- v0.3.1 add examples for both 2D denoising and 3D denoising, and integrate them into the SN2Nexecute function.
- v0.2.8 reorder the core code
- v0.1.0 initial version

## &#x1F308; Resources: 

- **Some fancy results and comparisons:** [Lab's website](https://weisongzhao.github.io/home/portfolio-4-col.html#SN2N)
- **Preprint:** [Liying Qu et al. Self-inspired learning to denoise for live-cell super-resolution microscopy, bioRxiv (2024).](https://doi.org/10.1101/2024.01.23.576521)
- **Percentile normalization plugin:** https://github.com/WeisongZhao/SN2N


## Open source [SN2N](https://github.com/WeisongZhao/SN2N)
This software and corresponding methods can only be used for **non-commercial** use, and they are under Open Data Commons Open Database License v1.0.