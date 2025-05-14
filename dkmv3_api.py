import os, sys
import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from time import time
current_path = os.path.dirname(os.path.abspath(__file__))

from dkm import DKMv3_indoor, DKMv3_outdoor
import matplotlib.pyplot as plt

import pdb

from time import time


px1 = None
def get_flow_api():

    
    device = 'cuda:0'
    dkm_model = DKMv3_indoor(device=device)
    dkm_model.eval()

    #dkm_model2 = DKMv3_outdoor(device=device)
    #dkm_model2.eval()

    s = 1
    def fn_indoor(image1, image2, inference_size=(480, 640)):
        '''
            image1, image2: H,W,3
        '''

        ori_size = image1.shape[:2]
        image1 = Image.fromarray(image1)
        image2 = Image.fromarray(image2)

        if inference_size is not None:
            if ori_size[0] != inference_size[0] or ori_size[1] != inference_size[1]:
                H,W = inference_size
                image1 = image1.resize((W,H))
                image2 = image2.resize((W,H))
            else:
                H,W = ori_size

        else:
            H,W = ori_size
        warp, certainty = dkm_model.match(image1, image2, device=device) # warp is H,2W,4
        px1 = warp[:,:W,:2]
        px2 = warp[:,:W,2:]
        px2[...,0] = W/2*(px2[...,0]+1)
        px2[...,1] = H/2*(px2[...,1]+1)
        px1[...,0] = W/2*(px1[...,0]+1)
        px1[...,1] = H/2*(px1[...,1]+1)

        flow_ = px2 - px1 
        flow_ = torch.permute(flow_,(2,0,1))[None]

        flow = flow_

        # flow_back
        px1 = warp[:,W:,:2]
        px2 = warp[:,W:,2:]
        px2[...,0] = W/2*(px2[...,0]+1)
        px2[...,1] = H/2*(px2[...,1]+1)
        px1[...,0] = W/2*(px1[...,0]+1)
        px1[...,1] = H/2*(px1[...,1]+1)

        flow_back_ = px1 - px2
        flow_back_ = torch.permute(flow_back_,(2,0,1))[None]

        flow_back = flow_back_

        certainty_back = certainty[:,W:]
        certainty = certainty[:,:W]
        if inference_size is not None:
            if ori_size[0] != inference_size[0] or ori_size[1] != inference_size[1]:
                flow_pr = F.interpolate(flow, size=ori_size, mode='bilinear',
                                            align_corners=False)
                flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
                flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

                certainty = F.interpolate(certainty[None,None], size=ori_size, mode='bilinear',
                                            align_corners=False)
                certainty_back = F.interpolate(certainty_back[None,None], size=ori_size, mode='bilinear',
                                            align_corners=False)

                flow = flow_pr




                flow_back_pr = F.interpolate(flow_back, size=ori_size, mode='bilinear',
                                            align_corners=False)
                flow_back_pr[:, 0] = flow_back_pr[:, 0] * ori_size[-1] / inference_size[-1]
                flow_back_pr[:, 1] = flow_back_pr[:, 1] * ori_size[-2] / inference_size[-2]

                flow_back = flow_back_pr

                '''

                flow_pr = cv2.resize(flow, size=(ori_size[-1],ori_size[-2]), interpolation=cv2.INTER_AREA)
                flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
                flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

                flow = flow_pr
                '''

        return flow[0], flow_back[0], certainty, certainty_back

    def fn_outdoor(image1, image2, inference_size=(864, 1152)):
        '''
            image1, image2: H,W,3
        '''

        ori_size = image1.shape[:2]
        image1 = Image.fromarray(image1)
        image2 = Image.fromarray(image2)

        if inference_size is not None:
            if ori_size[0] != inference_size[0] or ori_size[1] != inference_size[1]:
                H,W = inference_size
                image1 = image1.resize((W,H))
                image2 = image2.resize((W,H))
            else:
                H,W = ori_size

        else:
            H,W = ori_size
        warp, certainty = dkm_model2.match(image1, image2, device=device) # warp is H,2W,4
        px1 = warp[:,:W,:2]
        px2 = warp[:,:W,2:]
        px2[...,0] = W/2*(px2[...,0]+1)
        px2[...,1] = H/2*(px2[...,1]+1)
        px1[...,0] = W/2*(px1[...,0]+1)
        px1[...,1] = H/2*(px1[...,1]+1)

        flow_ = px2 - px1 
        flow_ = torch.permute(flow_,(2,0,1))[None]

        flow = flow_

        # flow_back
        px1 = warp[:,W:,:2]
        px2 = warp[:,W:,2:]
        px2[...,0] = W/2*(px2[...,0]+1)
        px2[...,1] = H/2*(px2[...,1]+1)
        px1[...,0] = W/2*(px1[...,0]+1)
        px1[...,1] = H/2*(px1[...,1]+1)

        flow_back_ = px1 - px2
        flow_back_ = torch.permute(flow_back_,(2,0,1))[None]

        flow_back = flow_back_

        certainty_back = certainty[:,W:]
        certainty = certainty[:,:W]
        if inference_size is not None:
            if ori_size[0] != inference_size[0] or ori_size[1] != inference_size[1]:
                flow_pr = F.interpolate(flow, size=ori_size, mode='bilinear',
                                            align_corners=False)
                flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
                flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

                certainty = F.interpolate(certainty[None,None], size=ori_size, mode='bilinear',
                                            align_corners=False)
                certainty_back = F.interpolate(certainty_back[None,None], size=ori_size, mode='bilinear',
                                            align_corners=False)


                flow = flow_pr




                flow_back_pr = F.interpolate(flow_back, size=ori_size, mode='bilinear',
                                            align_corners=False)
                flow_back_pr[:, 0] = flow_back_pr[:, 0] * ori_size[-1] / inference_size[-1]
                flow_back_pr[:, 1] = flow_back_pr[:, 1] * ori_size[-2] / inference_size[-2]

                flow_back = flow_back_pr

                '''

                flow_pr = cv2.resize(flow, size=(ori_size[-1],ori_size[-2]), interpolation=cv2.INTER_AREA)
                flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
                flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

                flow = flow_pr
                '''

        return flow[0], flow_back[0], certainty, certainty_back
    def fn(image1, image2):
        flow1, flow_back1, c1, cb1 = fn_indoor(image1, image2)
        return flow1, flow_back1, c1, cb1
        '''
        # only kitti use outdoor
        flow2, flow_back2, c2, cb2 = fn_outdoor(image1, image2)
        return (flow1+flow2)/2, (flow_back1+flow_back2)/2, (c1+c2)/2, (cb1+cb2)/2
        '''
    return fn

if __name__ == '__main__':
    im1 = cv2.imread('/home/yijun/data/Replica/office0/results/frame000000.jpg',-1) 
    im2 = cv2.imread('/home/yijun/data/Replica/office0/results/frame000050.jpg',-1)
    f_dkm = get_flow_api()
    pdb.set_trace()
    flow, certainty = f_dkm(im1, im2) 

