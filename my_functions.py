from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models

# general libs
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
import argparse
import copy


### My libs
from dataset.dataset import DAVIS_MO_Test
from model.model import STM

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def warp_image(image, flow):
    """
    Warps an image with the given flow.
    """
    B, C, H, W = image.size()
    # print(str(B) + " " + str(C) + " " + str(H) + " " + str(W))

    # Generate mesh grid.
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    flow = torch.from_numpy(flow).unsqueeze(0).permute(0, 3, 1, 2)

    if image.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flow

    # Normalize to -1,1
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:] / max(W-1,1) - 1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:] / max(H-1,1) - 1.0

    warped_image = F.grid_sample(image, vgrid.permute(0,2,3,1), mode='bilinear', padding_mode='zeros')
    return warped_image

def load_image_as_tensor(img):

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert the NumPy array to a PyTorch tensor and reshape it
    img_tensor = torch.FloatTensor(np.ascontiguousarray(img.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))

    # Add a batch dimension if needed (for a single image, make it a batch of 1)
    img_tensor = img_tensor.unsqueeze(0)  # Shape becomes (1, C, H, W)
    return img_tensor

def generate_warped_image(img):
    warped_image_cpu = img.cpu()
    warped_image_cpu = warped_image_cpu * 255
    warped_image_np = warped_image_cpu.numpy().astype('uint8')

    warped_image_np = warped_image_np.transpose((1, 2, 0))

    return warped_image_np

def warp_mask_with_flow(mask, flow):
    """
    Warp a mask using an optical flow field.
    Args:
        mask (Tensor): The mask to be warped, size ([1, 11, 1, 480, 854])
        flow (Tensor): The optical flow result, size ([2, 480, 854])
    Returns:
        Tensor: The warped mask, size ([1, 11, 1, 480, 854])
    """
    # Ensure the flow is in the shape expected by grid_sample, which is (N, H, W, 2)
    B, C, _, H, W = mask.shape
    device = mask.device
    
    # Create coordinate grid for the original mask
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H, device=device), 
                                    torch.arange(0, W, device=device), indexing='ij')
    grid = torch.stack((grid_x, grid_y), 2).float()  # Shape: (H, W, 2)
    grid = grid.unsqueeze(0)  # Shape: (1, H, W, 2)
    grid = grid.repeat(B, 1, 1, 1)  # Shape: (B, H, W, 2)

    # Adjust grid by optical flow
    flow = flow.unsqueeze(0)  # Adding a batch dimension for consistency, Shape: (1, 2, H, W)
    flow = flow.permute(0, 2, 3, 1)  # Reorder to (N, H, W, 2)
    grid = grid + flow  # New locations to sample from

    # Normalize grid to [-1, 1] to match grid_sample's expected input
    grid[:, :, :, 0] = 2.0 * grid[:, :, :, 0] / max(W - 1, 1) - 1.0
    grid[:, :, :, 1] = 2.0 * grid[:, :, :, 1] / max(H - 1, 1) - 1.0

    # Warp the mask using the adjusted grid
    mask = mask.squeeze(2)  # Remove the singleton dimension for grid_sample compatibility
    warped_mask = F.grid_sample(mask, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    warped_mask = warped_mask.unsqueeze(2)  # Restore the singleton dimension

    return warped_mask