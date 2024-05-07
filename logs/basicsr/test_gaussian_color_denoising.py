# ------------------------------------------------------------------------
# Modified from Restormer (https://github.com/swz30/Restormer)
# ------------------------------------------------------------------------

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F

from basicsr.models.archs.CGNet_Guassian_arch import CascadedGazeNetBigger, CascadedGazeNetBiggerLocal
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
from pdb import set_trace as stx
import math
import cv2

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader



def calculate_psnr(img1, img2, border=0):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))




parser = argparse.ArgumentParser(description='Gaussian Color Denoising using Restormer')
parser.add_argument('--input_dir', default='path_toguassian_noise test_datasets/datasets/guassian_noise/test/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir_tmp', default='path_to_save_photos', type=str, help='Directory for results')
parser.add_argument('--sigmas', default='15', type=str, help='Sigma values')
args = parser.parse_args()


yaml_file = "path to the train yml file"
weights = "path to the trained model weights"

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)
s = x['network_g'].pop('type')

checkpoint = torch.load(weights)
model_restoration = CascadedGazeNetBiggerLocal(**x['network_g'])
model_restoration.load_state_dict(checkpoint['params'])


##########################
sigmas = np.int_(args.sigmas.split(','))
factor = 8
datasets = ['CBSD68','Kodak', 'McMaster', 'Urban100']


print(f"Experiment: { yaml_file.split('/')[-1]}")
for sigma_test in sigmas:
    print("Compute results for noise level",sigma_test)
    model_restoration.cuda()
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()
    
    for dataset in datasets:
        PSNR_List = []
        inp_dir = os.path.join(args.input_dir, dataset)
        # result_dir_tmp = os.path.join(result_dir_tmp, dataset, str(sigma_test))
        # os.makedirs(result_dir_tmp, exist_ok=True)
        files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.tif')))
        with torch.no_grad():
            for file_ in tqdm(files):
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
                img = np.float32(load_img(file_))/255.
                img_copy = np.float32(load_img(file_))

                np.random.seed(seed=80)  # for reproducibility
                img += np.random.normal(0, sigma_test/255., img.shape)

                img = torch.from_numpy(img).permute(2,0,1)
                input_ = img.unsqueeze(0).cuda()

                h,w = input_.shape[2], input_.shape[3]
                restored = model_restoration(input_)

                restored = restored[:,:,:h,:w]

                restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
                noisy_save = torch.clamp(input_,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

                psnr = calculate_psnr(restored*255.0, img_copy)
                PSNR_List.append(psnr)

                # save_file = os.path.join(result_dir_tmp, os.path.split(file_)[-1])
                # save_file_noise = os.path.join(result_dir_tmp, "noisy_"+os.path.split(file_)[-1])
                # save_img(save_file, img_as_ubyte(restored))
                # save_img(save_file, img_as_ubyte(noisy_save))

        print(f'PSNR value for {dataset} and sigma = {sigma_test} is = {np.mean(PSNR_List)}')