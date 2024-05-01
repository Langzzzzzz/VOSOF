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
import pandas as pd

### My libs
from dataset.dataset import DAVIS_MO_Test
from model.model import STM
from model.FusionNet import FusionNet

from pytorch_pwc.run import calcOpticalFlow
from my_functions import warp_image, generate_warped_image, warp_mask_with_flow

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from evaldavis2017.davis2017.davis import DAVIS
from evaldavis2017.davis2017.metrics import db_eval_boundary, db_eval_iou
from evaldavis2017.davis2017 import utils
from evaldavis2017.davis2017.results import Results
from scipy.optimize import linear_sum_assignment

import warnings
warnings.filterwarnings("ignore")

def Run_video(dataset,video, num_frames, num_objects,model,Mem_every=None, Mem_number=None):
    # initialize storage tensors
    if Mem_every:
        to_memorize = [int(i) for i in np.arange(0, num_frames, step=Mem_every)]
    elif Mem_number:
        to_memorize = [int(round(i)) for i in np.linspace(0, num_frames, num=Mem_number+2)[:-1]]
    else:
        raise NotImplementedError
    F_last,M_last, num_objects, _ = dataset.load_single_image(video,0)
    '''
    F_last size:  torch.Size([3, 1, 480, 854])  M_last size:  torch.Size([11, 1, 480, 854])
    after unsqueeze(0) - F_last size:  torch.Size([1, 3, 1, 480, 854])  M_last size:  torch.Size([1, 11, 1, 480, 854])
    '''
    F_last = F_last.unsqueeze(0)
    M_last = M_last.unsqueeze(0)
    E_last = M_last
    pred = np.zeros((num_frames,M_last.shape[3],M_last.shape[4]))
    all_Ms = []
    for t in range(1,num_frames):

        # memorize
        with torch.no_grad():
            prev_key, prev_value = model(F_last[:,:,0], E_last[:,:,0], torch.tensor([num_objects])) 

        if t-1 == 0: # 
            this_keys, this_values = prev_key, prev_value # only prev memory
        else:
            this_keys = torch.cat([keys, prev_key], dim=3)
            this_values = torch.cat([values, prev_value], dim=3)
        del prev_key,prev_value

        F_,M_, num_objects, _ = dataset.load_single_image(video,t)

        F_ = F_.unsqueeze(0)
        M_ = M_.unsqueeze(0)
        all_Ms.append(M_.cpu().numpy())
        # segment
        with torch.no_grad():
            logit = model(F_[:,:,0], this_keys, this_values, torch.tensor([num_objects])) # logit shape:  torch.Size([1, 11, 480, 854])
        E = F.softmax(logit, dim=1)
        softmax_M = F.softmax(M_, dim=1)
        del logit
        # update

        if t-1 in to_memorize:
            keys, values = this_keys, this_values
            del this_keys,this_values
        
        # pred[t] = torch.argmax(E[0], dim=0).cpu().numpy().astype(np.uint8)

        '''
        combine with optical flow result
        '''
        # step 1 get optical flow result
        curr_frame = torch.FloatTensor(F_.cpu().numpy().squeeze())
        last_frame = torch.FloatTensor(F_last.cpu().numpy().squeeze())
        OF = calcOpticalFlow(curr_frame, last_frame) # OF shape:  (480, 854, 2)

        # step 2 warp the last frame
        warped_curr_frame = warp_image(last_frame.unsqueeze(0), OF).squeeze(0) # torch.Size([3, 480, 854])
        # warped_curr_frame_img = generate_warped_image(warped_curr_frame) # numpy (480, 854, 3)

        # step 3 calculate the photometric loss 
        l2_photometric_loss_map = ((warped_curr_frame - curr_frame) ** 2).sum(dim=0)
        max_possible_squared_difference = (1**2) * warped_curr_frame.shape[0]  # Max squared difference for images in [0, 1]
        normalized_photometric_loss_map = l2_photometric_loss_map / max_possible_squared_difference # photometric loss shape:  torch.Size([480, 854])
        normalized_photometric_loss_map = 1 - normalized_photometric_loss_map

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # get warpped mask of image 2
        warped_curr_Mask = warp_mask_with_flow(E_last.to(device), torch.from_numpy(OF).permute(2,0,1).to(device)) # torch.Size([1, 11, 480, 854])
        # calculate the confidence score of warpped mask 
        confidence_scores = normalized_photometric_loss_map.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        OF_confidence_score = confidence_scores.to(device) * warped_curr_Mask.to(device)
        estimate_curr_E = OF_confidence_score.squeeze(2) # torch.Size([1, 11, 480, 854])

        # combine the OF_confidence_score and the segmentation result        
        threshold = torch.mean(estimate_curr_E).item()  # Define a confidence threshold
        mask = estimate_curr_E > threshold  # Create a mask for high-confidence areas
        combined_score = torch.where(mask, estimate_curr_E, E)
        pred[t] = torch.argmax(combined_score[0], dim=0).cpu().numpy().astype(np.uint8)
        
        E_last = E.unsqueeze(2) # torch.Size([1, 11, 1, 480, 854])
        F_last = F_
        del M_

    Ms = np.concatenate(all_Ms,axis=2)
    return pred,Ms

def evaluate_semisupervised(all_gt_masks, all_res_masks, all_void_masks, metric):
    if all_res_masks.shape[0] > all_gt_masks.shape[0]:
        sys.stdout.write("\nIn your PNG files there is an index higher than the number of objects in the sequence!")
        sys.exit()
    elif all_res_masks.shape[0] < all_gt_masks.shape[0]:
        zero_padding = np.zeros((all_gt_masks.shape[0] - all_res_masks.shape[0], *all_res_masks.shape[1:]))
        all_res_masks = np.concatenate([all_res_masks, zero_padding], axis=0)
    j_metrics_res, f_metrics_res = np.zeros(all_gt_masks.shape[:2]), np.zeros(all_gt_masks.shape[:2])
    for ii in range(all_gt_masks.shape[0]):
        if 'J' in metric:
            j_metrics_res[ii, :] = db_eval_iou(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
        if 'F' in metric:
            f_metrics_res[ii, :] = db_eval_boundary(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
    return j_metrics_res, f_metrics_res

def evaluate(model,Testloader,metric):

    # Containers
    metrics_res = {}
    if 'J' in metric:
        metrics_res['J'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
    if 'F' in metric:
        metrics_res['F'] = {"M": [], "R": [], "D": [], "M_per_object": {}}

    for V in tqdm.tqdm(Testloader):
        print("Line 112 - V: ", V)
        num_objects, info = V
        seq_name = info['name']
        num_frames = info['num_frames']
        print("Line 118 - num_objects: ", num_objects, " size: ", str(num_objects.size()))
        print('[{}]: num_frames: {}, num_objects: {}'.format(seq_name, num_frames, num_objects[0]))
        
        pred,Ms = Run_video(Testloader, seq_name, num_frames, num_objects,model,Mem_every=5, Mem_number=None)

        # line 130 pred shape:  (80, 480, 854) Ms shape:  (1, 11, 79, 480, 854)
        # print("line 130 pred shape: ", pred.shape, "Ms shape: ", Ms.shape)
        # all_res_masks = Es[0].cpu().numpy()[1:1+num_objects]
        all_res_masks = np.zeros((num_objects,pred.shape[0],pred.shape[1],pred.shape[2]))
        for i in range(1,num_objects+1):
            all_res_masks[i-1,:,:,:] = (pred == i).astype(np.uint8)
        all_res_masks = all_res_masks[:, 1:-1, :, :]
        all_gt_masks = Ms[0][1:1+num_objects]
        all_gt_masks = all_gt_masks[:, :-1, :, :]
        j_metrics_res, f_metrics_res = evaluate_semisupervised(all_gt_masks, all_res_masks, None, metric)
        for ii in range(all_gt_masks.shape[0]):
            if 'J' in metric:
                [JM, JR, JD] = utils.db_statistics(j_metrics_res[ii])
                metrics_res['J']["M"].append(JM)
                metrics_res['J']["R"].append(JR)
                metrics_res['J']["D"].append(JD)
            if 'F' in metric:
                [FM, FR, FD] = utils.db_statistics(f_metrics_res[ii])
                metrics_res['F']["M"].append(FM)
                metrics_res['F']["R"].append(FR)
                metrics_res['F']["D"].append(FD)

    # df.to_csv('train_dataset.csv', index=False)
    J, F = metrics_res['J'], metrics_res['F']
    g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
    final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
    g_res = np.array([final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
                      np.mean(F["D"])])
    return g_res
	    



if __name__ == "__main__":
    torch.set_grad_enabled(False) # Volatile
    def get_arguments():
        parser = argparse.ArgumentParser(description="xxx")
        parser.add_argument("-g", type=str, help="0; 0,1; 0,3; etc", required=True)
        parser.add_argument("-s", type=str, help="set", required=True)
        parser.add_argument("-y", type=int, help="year", required=True)
        parser.add_argument("-D", type=str, help="path to data",default='/root/autodl-tmp/data/DAVIS/2017/trainval')
        parser.add_argument("-backbone", type=str, help="backbone ['resnet50', 'resnet18','resnest101']",default='resnet50')
        parser.add_argument("-p", type=str, help="path to weights",default='/root/autodl-tmp/code/Training-Code-of-STM/davis_youtube_resnet50_799999_170.pth')
        return parser.parse_args()

    args = get_arguments()

    GPU = args.g
    YEAR = args.y
    SET = args.s
    DATA_ROOT = args.D

    # Model and version
    MODEL = 'STM'
    print(MODEL, ': Testing on DAVIS')

    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    if torch.cuda.is_available():
        print('using Cuda devices, num:', torch.cuda.device_count())

    Testloader = DAVIS_MO_Test(DATA_ROOT, resolution='480p', imset='20{}/{}.txt'.format(YEAR,SET), single_object=True)
    model = nn.DataParallel(STM(args.backbone))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    pth = args.p

    model.load_state_dict(torch.load(pth))
    metric = ['J','F']
    print(evaluate(model,Testloader,metric))
