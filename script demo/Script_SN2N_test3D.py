# -*- coding: utf-8 -*-

import os
import torch
import tifffile as tiff
import argparse
import numpy as np
import itertools
import tqdm

def normalize_(input):
    img = np.array(input).astype('float32')
    img = img - np.min(img)
    img = img / np.max(img)
    return img


def TOTENSOR_(img_datas_np):
    if len(img_datas_np.shape) == 2:
       img_datas_np = np.expand_dims(img_datas_np, axis = 0)
       img_datas_np = np.expand_dims(img_datas_np, axis = 0)
    return torch.from_numpy(img_datas_np).type(torch.FloatTensor)

def apply(model, data, device, overlap_shape=None, verbose=False):
    '''
    Applies a model to an input image. The input image stack is split into
    sub-blocks with model's input size, then the model is applied block by
    block. The sizes of input and output images are assumed to be the same
    while they can have different numbers of channels.

    Parameters
    ----------
    model: keras.Model
        Keras model.
    data: array_like or list of array_like
        Input data. Either an image or a list of images.
    overlap_shape: tuple of int or None
        Overlap size between sub-blocks in each dimension. If not specified,
        a default size ((32, 32) for 2D and (2, 32, 32) for 3D) is used.
        Results at overlapped areas are blended together linearly.

    Returns
    -------
    ndarray
        Result image.
    '''

    model_input_image_shape = (16, 640, 640)
    model_output_image_shape = (16, 640, 640)

    if len(model_input_image_shape) != len(model_output_image_shape):
        raise NotImplementedError

    image_dim = len(model_input_image_shape)
    num_input_channels = 1
    num_output_channels = 1


    if overlap_shape is None:
        if image_dim == 2:
            overlap_shape = (32, 32)
        elif image_dim == 3:
            overlap_shape = (2, 32, 32)
        else:
            raise NotImplementedError
    elif len(overlap_shape) != image_dim:
        raise ValueError(f'Overlap shape must be {image_dim}D; '
                         f'Received shape: {overlap_shape}')

    step_shape = tuple(
        m - o for m, o in zip(
            model_input_image_shape, overlap_shape))

    block_weight = np.ones(
        [m - 2 * o for m, o
         in zip(model_output_image_shape,overlap_shape)],
        dtype=np.float32)

    block_weight = np.pad(
        block_weight,
        [(o + 1, o + 1) for o in overlap_shape],
        'linear_ramp'
    )[(slice(1, -1),) * image_dim]

    batch_size = 1
    batch = np.zeros(
        (batch_size, num_input_channels, *model_input_image_shape),
        dtype=np.float32)

    if isinstance(data, (list, tuple)):
        input_is_list = True
    else:
        data = [data]
        input_is_list = False

    result = []

    for image in data:
        input_image_shape = image.shape
        output_image_shape = input_image_shape

        applied = np.zeros(
            (output_image_shape), dtype=np.float32)
        sum_weight = np.zeros(output_image_shape, dtype=np.float32)

        num_steps = tuple(
            i // s + (i % s != 0)
            for i, s in zip(input_image_shape, step_shape))

        # top-left corner of each block
        blocks = list(itertools.product(
            *[np.arange(n) * s for n, s in zip(num_steps, step_shape)]))

        for chunk_index in tqdm.trange(
                0, len(blocks), batch_size, disable=not verbose,
                dynamic_ncols=True, ascii=tqdm.utils.IS_WIN):
            rois = []
            for batch_index, tl in enumerate(
                    blocks[chunk_index:chunk_index + batch_size]):
                br = [min(t + m, i) for t, m, i
                      in zip(tl, model_input_image_shape, input_image_shape)]
                r1, r2 = zip(
                    *[(slice(s, e), slice(0, e - s)) for s, e in zip(tl, br)])

                m = image[r1]
                if model_input_image_shape != m.shape:
                    pad_width = [(0, b - s)for b, s
                                  in zip(model_input_image_shape, m.shape)]
                    pad_width = np.array(pad_width, dtype = 'int')
                    m = np.pad(m, pad_width, 'reflect')
                    
                    
                    
                (t, x, y) = m.shape
                batch = np.zeros(
                    (batch_size, num_input_channels, t, x, y),
                    dtype=np.float32)
                batch[batch_index] = m
                rois.append((r1, r2))

            p = np.zeros((batch.shape))
            datatensor = TOTENSOR_(batch)
            test_pred = model(datatensor.to(device))#小块不用归一化
            test_pred = test_pred.to(torch.device("cpu"))
            p[:, :, :, :, :] = test_pred.detach().numpy()
            
            
            
            for batch_index in range(len(rois)):
                for channel in range(num_output_channels):
                    p[batch_index, channel, ...] *= block_weight

                r1, r2 = [roi for roi in rois[batch_index]]
                applied[r1] += p[batch_index, channel][r2]
                sum_weight[r1] += block_weight[r2]

        for channel in range(num_output_channels):
            applied[...] /= sum_weight

        applied = 255*normalize_(applied)
        result.append(applied)

    return result if input_is_list else result[0]


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = args.model_path
    model = torch.load(model_path, map_location=device)
    model = model.to(device)

    img_path = args.img_path
    save_path = args.save_path
    epoch = args.epoch
    overlap_shape = tuple(map(int, args.overlap_shape.split(',')))

    with torch.no_grad():
        filenames = []
        for (root, dirs, files) in os.walk(img_path):
            for j, Ufile in enumerate(files):
                imgpath = os.path.join(root, Ufile)
                input_tif = tiff.imread(imgpath)

                input_tif = np.squeeze(input_tif).astype('uint8')
                result = apply(model, input_tif, device, overlap_shape=overlap_shape)
                os.makedirs(save_path, exist_ok=True)
                tiff.imwrite(f'{save_path}/{Ufile}_predict_epoch_{epoch}.tif', result.astype('uint8'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image Application Script")
    parser.add_argument('--cuda_device', type=str, default="0", help='CUDA device ID')
    parser.add_argument('--img_path', type=str, required=True, help='Path to the input images')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the predictions')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--epoch', type=int, default=100, help='Epoch number for prediction')
    parser.add_argument('--overlap_shape', type=str, default="2,256,256", help='Overlap shape in format H,W (e.g., 2,256,256)')

    args = parser.parse_args()
    main(args)
