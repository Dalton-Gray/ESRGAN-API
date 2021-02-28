import numpy as np
import math
import os
import ntpath
import cv2
from skimage.measure import compare_ssim as ssim
from matplotlib import pyplot as plt
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.optimizers import Adam
import torch 


# Not Finished 

def mse(target, ref):

    # convert images to numpy array
    # target = target.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    # ref = ref.data.squeeze().float().cpu().clamp_(0, 1).numpy() 

    # err = np.sum((target.float() - ref.astype('float')) ** 2)
    # err /= float(target.shape[0] * target.shape[1])
    err = 0
    return err

def compare_images(target, ref):

    scores = []
    # scores.append(psnr(target, ref))
    scores.append(mse(target, ref))
    # scores.append(ssim(target, ref, multichannel = True))
    
    return scores
