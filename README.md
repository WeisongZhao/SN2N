
[![Github commit](https://img.shields.io/github/last-commit/WeisongZhao/SN2N)](https://github.com/WeisongZhao/SN2N/)
[![Github All Releases](https://img.shields.io/github/downloads/WeisongZhao/SN2N/total.svg)](https://github.com/WeisongZhao/SN2N/releases/tag/v0.1.0/)
[![License](https://img.shields.io/github/license/WeisongZhao/SN2N)](https://github.com/WeisongZhao/SN2N/blob/master/LICENSE/)
[![paper](https://img.shields.io/badge/paper-nat.%20methods.-black.svg)](https://www.nature.com/nmeth/)
[![releases](https://img.shields.io/badge/release-v0.1.0-FF6600.svg)](https://github.com/WeisongZhao/SN2N/releases/tag/v0.1.0/)<br>

[![Twitter](https://img.shields.io/twitter/follow/QuLiying?label=liying)](https://twitter.com/weisong_zhao)
[![Twitter](https://img.shields.io/twitter/follow/weisong_zhao?label=weisong)](https://twitter.com/QuLiying)
[![GitHub stars](https://img.shields.io/github/stars/WeisongZhao/SN2N?style=social)](https://github.com/WeisongZhao/SN2N/) 



<p>
<h1 align="center"><font color="#FF6600">S</font>N2N</h1>
<h5 align="center">Self-inspired learning to denoise for live-cell super-resolution microscopy.</h5>
<h6 align="right">v0.1.0</h6>
</p>
<br>


<p>
<img src='./imgs/4color-small.gif' align="left" width=190>
</p>
<br>


This repository is for our developed self-inspired Noise2Noise (SN2N) learning-to-denoise engine, and it will be in continued development. It is distributed as accompanying software for publication: [Liying Qu et al. Self-inspired learning to denoise for live-cell super-resolution microscopy, XXX (2024)](https://www.nature.com/nmeth/). Please cite SN2N in your publications, if it helps your research.

<br><br><br>

<div align="center">

✨ [**Introduction**](#-Introduction) **|**  🔧 [**Install**](#-Install)  **|** 🎨 [**Data generation**](#-Data_generation) **|**  💻 [**Training**](#-Training) **|** ⚡ [**Inference**](#-Inference)**|** &#x1F308; [**Results**](#-Results)

</div>

---

## Introduction

Our SN2N is fully competitive with the supervised learning methods and overcomes the need for large dataset and clean ground-truth. **First**, we create a self-supervised data generation strategy based on super-resolution images' spatial redundancy, using a diagonal resampling step followed by a Fourier interpolation for single-frame Noise2Noise. **Second**, we have taken a step further by ushering in a self-constrained learning process to enhance the performance and data-efficiency. **Finally**, we develop random patch transformations in multiple dimensions (Patch2Patch) to further improve the data efficiency.Patch2Patch can equivalently create more imaging results without changing the inherent noise properties and hence it can effectively reduce the required data bulk. Detailed workflow of SN2N can be seen as follows:

<p align="center">
  <img src="SN2N-workflow.png" width='600'>
</p> 


## 🔧 Install

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

    ```bash
    $ conda create -n SN2N python=3.7.6
    $ conda activate SN2N
    $ pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
    $ pip install tifffile numpy scipy scikit-image scikit-image tqdm
    ```

## 🎨Data generation

### 1. Self-supervised data generation
Our SN2N is adaptable to both 2D (**xy**) and 3D (**xyz**) datasets. You can use your own data or our uploaded to generate 2D / 3D SN2N data pairs.

#### 2D data generation
```bash
	# Data generation
    python scripts_sn2n_datagenerate_2D.py --train_data_path "noisy" --name "simu_2D" --save_path "simu_2D_0213"
    
    #Key parameters:
 	--img_path
        path of raw images TO train
    --save_path
        path of patched dataset TO save
    --pre_augment_mode
    	Pa2ch2Patch augment mode (Optional)
        0: NONE; 
        1: direct interchange in t;
        2: interchange in single frame;
        3: interchange in multiple frame but in different regions
        {default: 0}
    --augment_mode
    	Basic augment mode (Optional)
        0: NONE; 
        1: double the dataset with random rotate&flip;
        2: eightfold the dataset with random rotate&flip;
        {default: 0}
    --img_res
        patch size
        {default: (128, 128)}
    --ifx2
        if re-scale TO original size
        True OR False
        {default: True}
    --inter_method
        Scaling method
        'Fourier': Fourier re-scaling;
        'bilinear': spatial re-scaling;
        {default: 'Fourier'}
    --threshold_mode (0 ~ 2)
        threshold mode to exclude some black patches
        {default: 2}
    --threshold (0 ~ 255)
        threshold to exclude some black patches
        {default: 15}
    --size
        ROI size of interchange in Patch2Patch
        {default: (64, 64)}
    --times
        Repeat times of ROI interchange in Patch2Patch
    --roll
        Repeat times of interchange for one image in Patch2Patch
```

#### 3D data generation
```bash
	# Data generation
    python scripts_sN2N_datagenerate_3D.py --train_data_path "noisy" --name "simu_3D" --save_path "simu_3D_0213"
    
    #Key parameters 3D: 	
    --multi_frame
        Number of Frames Used to Generate Three-Dimensional Training Data
```

## 💻 Training 

### 1. Prepare the data  

You can use SN2N data pairs for SN2N learning

### 2. Start training

#### 2D data training

```bash
    # training
    python Scripts_SN2N_train2D.py --cuda_device 0 --dataset_name "noisy_2D" --epoch 20 --reg 1 --prefix 0123_

    # Key parameters:  
    --dataset_name
    	Name of the dataset, used for training
    --epochs
    	Total number of training epochs
    --train_batch_size
    	how many patches will be extracted for training
    --test_batch_size
    	how many patches will be extracted for testing
    --reg
    	Weight of self-constrained loss
        {default: 1}
    --reg_sparse
     	Weight of sparse loss
        {default: 0}
	--prefix
		Prefix to add at the beginning When saving the model
	--ifadaptive_lr
		Whether to use adaptive learning rate.
        {default: False}
```

#### 3D data training

```bash
    # training
    python Scripts_SN2N_train3D.py --cuda_device 0 --dataset_name "noisy_3D" --epoch 20 --reg 1 --prefix 0123_
```


## ⚡ Inference
### 1. Prepare models

Before inference, you should have trained your own model

### 2. Test models
#### 2D inference

  ```bash
    # testing
    python Scripts_SN2N_test_2D.py --cuda_device 0 --img_path "path_to_images" --save_path "path_to_save_predictions" --model_path "path_to_model" --epoch 40 --ifGPU True

    # Key parameters:
    --cuda_device
    	CUDA device ID
    --imag_path
    	Path to the input images
    --save_path
    	Path to save the predictions
    --model_path
    	Path to the trained model
    --epoch
    	Epoch number of the model
    --ifGPU
    	Flag to use GPU or not
  ```

#### 3D inference
In 3D prediction tasks, we use the method of stitching predictions to avoid issues of memory overflow

  ```bash
    # tesing
    python Scripts_SN2N_test_3D.py --cuda_device 0 --img_path "path_to_images" --save_path "path_to_save_predictions" --model_path "path_to_model" --epoch 40 --ifGPU True --overlap_shape "2,256,256"

    # Key parameters:
    --overlap_shape
    	In the stitching prediction, the 'overlap' used
		default{2, 256,256}
  ```

## Version

- v0.1.0 initial version of SN2N

## Related links: 

- **Some fancy results and comparisons:** [my website](https://weisongzhao.github.io/MyWeb2/portfolio-4-col.html)
- **Preprint:** [Liying Qu et al. Self-inspired learning to denoise for live-cell super-resolution microscopy, bioRxiv (2024).]()


## Open source [SN2N](https://github.com/WeisongZhao/SN2N)
This software and corresponding methods can only be used for **non-commercial** use, and they are under Open Data Commons Open Database License v1.0.