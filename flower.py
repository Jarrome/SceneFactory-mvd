import os
import sys
import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import open3d as o3d
import matplotlib.pyplot as plt
from numpy.linalg import inv
from sklearn import linear_model
import scipy
from pyquaternion import Quaternion
from depth_cov.core.NonstationaryGpModule import NonstationaryGpModule
from depth_cov.utils.utils import sample_coords, normalize_coordinates


current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)
from dkmv3_api import get_flow_api
### Comment the early release huggingface version of v2
#from metric3d_api_hugface import get_mono_api
from metric3dv2_api import get_mono_api


from ext import remove_radius_outlier, estimate_normals, unproject_depth
from plot import depth23D, plot_depth_as_pc

import droid_ipf

from time import time
from icecream import ic
from rich import print

import gc
import pdb








class Flower:
    def __init__(self):
        self.inited = False
    def initialize(self, H, W, device, calib):
        self.H = H
        self.W = W
        s = 1
        self.s = s
        self.device = device
        self.calib = calib
        self.K = torch.eye(3).to(device)
        self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2] = calib 
        self.buffer = 21
        self.intrinsics = torch.as_tensor(self.calib, device="cuda", dtype=torch.float) // self.s
        self.disps_sens = torch.zeros(self.buffer, H//s, W//s, device="cuda", dtype=torch.float) # no observation


        grid_h, grid_w = torch.meshgrid(torch.range(0,H//s-1), torch.range(0,W//s-1), indexing='ij')

        self.px1 = np.stack([grid_w.numpy(), grid_h.numpy()]).reshape(2,-1)
        self.inited = True

        #self.uniMatch = get_flow_api()
        self.dkmv3 = get_flow_api()

        self.monodepth = get_mono_api()
        #self.dust3r = get_dust3r_api()

        #self.monodepth = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-large-hf")


    



    def estimate_dkmv3_flow(self, img1, img2):
        ''' unimatch
        img1 = torch.as_tensor(img1).permute(2, 0, 1).float()
        img2 = torch.as_tensor(img2).permute(2, 0, 1).float()
        flow = self.uniMatch(img1, img2, inference_size=(340, 600)).detach()#.cpu().numpy() 
        '''
        flow, flow_back, certainty, certainty_back = self.dkmv3(img1, img2)#.detach()#.cpu().numpy() 
        flow = flow.numpy() if not flow.is_cuda else flow.cpu().numpy()
        flow_back = flow_back.numpy() if not flow_back.is_cuda else flow_back.cpu().numpy()

        certainty = certainty.numpy() if not certainty.is_cuda else certainty.cpu().numpy()
 
        certainty_back = certainty_back.numpy() if not certainty_back.is_cuda else certainty_back.cpu().numpy()
        
       
        return flow, flow_back, certainty, certainty_back
        '''
        flow = self.uniMatch(img1, img2)
        return flow
        '''


    def estimate_flow(self, img1, img2):
        img1 = torch.as_tensor(img1).permute(2, 0, 1).float()
        img2 = torch.as_tensor(img2).permute(2, 0, 1).float()
        flow = self.uniMatch(img1, img2, inference_size=(340, 600)).detach().cpu().numpy() 
        #flow = flow.numpy() if not flow.is_cuda else flow.cpu().numpy()
        return flow

    def estimate_monodepth(self, img):
        H,W = img.shape[:2]
        torch.cuda.synchronize()
        st = time()
        #img_ = cv2.resize(img, (W//2, H//2))
        #depth = self.monodepth.infer_pil(img_)
        #depth = cv2.resize(depth, (W, H))
        depth, normal = self.monodepth(img, inference_size=(H, W), intrinsic=[v for v in self.calib])
        #depth = self.monodepth(img, inference_size=(H, W), intrinsic=[v for v in self.calib])
        #normal = None

        torch.cuda.synchronize()
        print('[bold pink1]monodepth take:', time()-st)
        return depth, normal
    def estimate_dust3r(self, img1, img2):
        depth = self.dust3r(img1, img2)
        return depth*10, None

        

    def mvs(self,frames,poses=None,intrinsics=None, cov_th=None, relax=False, depth_only=False, init_w_monodepth=True):
        depth, normal = self.estimate_monodepth(frames[0])
        depth = torch.as_tensor(depth).float()
        #normal = torch.as_tensor(normal).float()

        planar_eqs, planar_mask = None, None 
        if init_w_monodepth:
            pred_depth, o1, o2 = self.mvs_w_flow(frames, depth=depth.clone() if depth is not None else None, poses=poses, intrinsics=intrinsics, relax=relax, planar=(planar_eqs, planar_mask), depth_only=depth_only)
        else:
            pred_depth, o1, o2 = self.mvs_w_flow(frames, depth=None, poses=poses, intrinsics=intrinsics, relax=relax, planar=(planar_eqs, planar_mask), depth_only=depth_only)

        mono_info = (depth, normal)#.numpy().astype(np.float32), normal.numpy())




        return pred_depth, o1, o2, mono_info

        #return pred_depth_.cpu().numpy(), o1, o2
    

    def mvs_w_flow(self, frames, depth = None, cov_th = None, poses=None, intrinsics=None, relax=False, planar=None, depth_only=False):
        mono_init = poses is None
        baseline_th = .05 if relax else .1
 
        mid = 0 #(len(frames)-1)//2
        nb = len(frames)-1
        ii = torch.as_tensor([0]*nb, dtype=torch.long, device=self.device)
        jj = torch.as_tensor([*range(mid+1,mid+nb+1)], dtype=torch.long, device=self.device)
        st = time()
        pxs = []
        ws = []

        H,W,s = self.H, self.W,self.s
        bd = 0 #min(H,W)//8
        # the out of bound mask
        w_ = np.ones_like(self.px1)

        # the mask
        for i in range(nb+1):
            if i != mid:
                st_ = time()
                flow, flow_back, certainty, certainty_back = self.estimate_dkmv3_flow(frames[mid], frames[i])
                flow = flow.reshape(2,-1)
                flow_back = flow_back.reshape(2,-1)
                torch.cuda.empty_cache()
                px = self.px1 + flow
                # check coverage
                w = np.ones_like(px)

                if True:#not depth_only:
                    w[:,px[0,:]>W//s-1-bd] = 0
                    w[:,px[0,:]<0+bd] = 0
                    w[:,px[1,:]>H//s-1-bd] = 0
                    w[:,px[1,:]<0+bd] = 0
                    # check consistency
                    px_ = px[:,w[0,:].astype(bool)]
                    grid = torch.as_tensor(px_.copy()).transpose(0,1).view(1,-1,1,2)
                    grid[:,:,:,0] -= (W-1)/2
                    grid[:,:,:,0] /= (W-1)/2
                    grid[:,:,:,1] -= (H-1)/2
                    grid[:,:,:,1] /= (H-1)/2
                    px1_ = px_ + torch.nn.functional.grid_sample(torch.as_tensor(flow_back.reshape((1,2,self.H,self.W))), grid, mode='bilinear', padding_mode='zeros', align_corners=True).reshape(2,-1).numpy()
                    diff = px1_ - self.px1[:,w[0,:].astype(bool)]
                    dist = (diff**2).sum(0)
                    print('[bold pink1]',dist.mean(),dist.std())
                    w[:,w[0,:]>0] *= (dist < (.5 * float(W) / 640)**2)[None,:]
                    if certainty is not None:
                        certainty[certainty<.1] = 0
                        certainty[(w[0].reshape(certainty.shape) == 0)] = 0
                        w *= certainty.reshape(1,-1)

                # check triangulation
                parallax_mask = self.check_flow_triangulation(flow, certainty)#, poses[i-1])
                w *= parallax_mask[None]

                pxs.append(px)
                ws.append(w)

                print('[bold pink1]flow',time()-st_)
        torch.cuda.synchronize()
        # all ws prod together
        nb = len(ws)
        w_ = (np.stack(ws)>0).sum(0) >= 1 #nb // 2 if nb >= 2 else (np.stack(ws)>0).sum(0) >0
        
        # BA for pose and inverse depth
        if depth_only: #False:#poses is not None:
            Tcws = [np.linalg.inv(pose) for pose in poses]
            #Tcws = [pose for pose in poses]

            qs = [Quaternion(matrix=Tcw, atol=1e-5,rtol=1e-5) for Tcw in Tcws]
            poses = torch.zeros(self.buffer, 7, device="cuda", dtype=torch.float)
            poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device="cuda")
            for i in range(nb):
                poses[i+1,:3] = torch.as_tensor(Tcws[i][:3,3]).to(poses)
                poses[i+1,3:6] = torch.as_tensor(qs[i].imaginary).to(poses)
                poses[i+1,6] = qs[i].real
        else:
            # droid's pose is Tcw but not Twc
            poses = torch.zeros(self.buffer, 7, device="cuda", dtype=torch.float)
            poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device="cuda")
        disps = torch.ones(self.buffer, H//s, W//s, device="cuda", dtype=torch.float) #/ 10
        if depth is not None:
            disps[mid] = torch.where(depth>0, 1.0/depth, 1.).cuda()


        target = np.stack(pxs) # B,2,HW
        target = torch.as_tensor(target).float().cuda().reshape(-1,2,H//s,W//s) # B, 2, H, W
        weights = np.stack(ws) # B,2,HW
        weight = torch.as_tensor(weights).float().cuda().reshape(-1,2,H//s,W//s) # B, 2, H, W


        t0 = 1 #max(1, ii.min()+1)
        t1 = max(ii.max(), jj.max())+1
        itrs = 20
        lm=1e-4
        ep=.1#0.1
        eta = 1e-6 * torch.ones((1, H//s, W//s)).to(disps) # for disps_sens
        st = time()
        motion_only=False
        if intrinsics is None:
            intrinsics = torch.as_tensor([W/2,W/2, (W-1)/2, (H-1)/2], device="cuda", dtype=torch.float) 
            opt_intr = True
        else:
            opt_intr = False
        cam_model_id = 0
        droid_ipf.ba(poses, disps, intrinsics, self.disps_sens,
            target, weight, eta, ii, jj, t0, t1, itrs, cam_model_id, lm, ep, motion_only, depth_only, opt_intr)


        print('[bold pink1]',poses[:nb+1,:3])
        
        
        print('[bold pink1]ba',time()-st)
        depth = 1/disps[mid]
    
        normal = None


        depth = depth.cpu().numpy()
        depth[depth < 0.001] = 0

        depth *= w_[0].reshape(depth.shape)
        # v2

        #depth_mean = depth.mean().item()
        #plt.imshow(depth.cpu().numpy().reshape(self.H,self.W)/(depth_mean+.01)*2)
        #plt.show()


        
        torch.cuda.synchronize()

        if s == 1:
            return depth, intrinsics.cpu().numpy(), poses.cpu().numpy()
        else:
            return cv2.resize(depth.cpu().numpy(), [self.W, self.H], interpolation = cv2.INTER_CUBIC),  cv2.resize(z_cov.cpu().numpy(), [self.W, self.H], interpolation = cv2.INTER_CUBIC)






    def get_good_neib_frames_v2(self, cur_pose, frames, nb_poses, k=2, cos_th=.866, epipole_th=1.2, baseline_th=.1, min_k=None, relaxed=False, nb_iskf=None):
        '''
            get the good neighbor frame for a frame
        '''
        center_pose = cur_pose
        inv_center_pose = np.linalg.inv(center_pose)

        # get relative pose to keyframe in window  
        poses = nb_poses #[frame.pose.matrix() for frame in frames]
        odos = [inv_center_pose@pose for pose in poses] #Twc


        # 1. threshold out large R
        # cos 30deg is 0.866, 60deg is 0.5
        ts = [odo[:3,(3,)] for odo in odos]

        z = np.array([[0,0,1]]).T #3x1
        angles = np.stack([(odo[:3,:3]@z)[2,0] for odo in odos])
        #baselines = np.stack([np.sqrt((t[:2]**2).sum()) for t in ts])
        baselines = np.stack([np.sqrt((t[:3]**2).sum()) for t in ts])
        baselines_xy = np.stack([np.sqrt((t[:2]**2).sum()) for t in ts])

        '''
        # 2. compute parallax score
        z = np.array([[0,0,2]]).T #3x1
        z_s = [odo[:3,:3]@z-z for odo in odos]
        z_s_ = np.stack(z_s)
        ts_ = np.stack(ts)
        '''

        # motivation is large baseline and facing to the same obj
        #score = ((z_s_ + ts_)**2).sum(1)[:,0] / baselines_xy
        
        valid = (angles > cos_th) * (baselines > baseline_th) # * (baselines_xy > baseline_th)  
        if nb_iskf is not None:
            valid *= nb_iskf

        if valid.sum() < k:
            return None, None
        

        # 2. compute parallax score
        z = np.array([[0,0,20]]).T #3x1
        z_s = [odo[:3,:3]@z+odo[:3,3:]-z for odo in odos]
        z_s_ = np.stack(z_s)
        ts_ = np.stack(ts)

        #score = 1/ (angles**5 * baselines_xy )#(z_s_**2).sum(1)[:,0]/((ts_**2).sum(1)[:,0]+1e-4)
        #score = (z_s_**2).sum(1)[:,0]#/((ts_**2).sum(1)[:,0]+1e-4)
        score = 1/((ts_**2).sum(1)[:,0]+1e-4)



        score[~valid] = np.inf

        ids_ = np.argsort(score)
        ids = []
        # only keep valid ids
        for id in ids_:
            if valid[id]:
                ids.append(id)

        #skip = len(ids)//k
        #ids = ids[::skip]
        
        nb_odos = [odos[id_] for id_ in ids[:k]]
        if frames is not None:
            nb_ims = [frame for frame in [frames[id_] for id_ in ids[:k]]]
        #
        return nb_ims, np.stack(nb_odos) #k,4,4



    def get_good_neib_frames(self, cur_pose, frames, nb_poses, k=2, cos_th=.866, epipole_th=1.2, baseline_th=.1, min_k=None, relaxed=False):
        '''
            get the good neighbor frame for a frame
        '''
        center_pose = cur_pose
        inv_center_pose = np.linalg.inv(center_pose)

        # get relative pose to keyframe in window  
        poses = nb_poses #[frame.pose.matrix() for frame in frames]
        odos = [inv_center_pose@pose for pose in poses]


        # 1. threshold out large R
        # cos 30deg is 0.866, 60deg is 0.5
        ts = [odo[:3,(3,)] for odo in odos]

        z = np.array([[0,0,1]]).T #3x1
        angles = np.stack([(odo[:3,:3]@z)[2,0] for odo in odos])
        #baselines = np.stack([np.sqrt((t[:2]**2).sum()) for t in ts])
        baselines = np.stack([np.sqrt((t[:3]**2).sum()) for t in ts])
        baselines_xy = np.stack([np.sqrt((t[:2]**2).sum()) for t in ts])


        # 1.    check epipole
        #       check angle
        #       check baselines
        epipole = np.stack(ts)[:,:,0]
        epipole[epipole[:,2]==0,2] = 1e-6
        epipole = epipole[:,:2] / epipole[:,(2,)]
        # compute back epipole
        odos_back = [np.linalg.inv(odo) for odo in odos]
        back_epipole = np.stack([odo[:3,(3,)] for odo in odos_back])[:,:,0] 
        back_epipole[back_epipole[:,2]==0,2] = 1e-6
        back_epipole = back_epipole[:,:2] / back_epipole[:, (2,)]

        # 2. compute parallax score
        z = np.array([[0,0,2]]).T #3x1
        z_s = [odo[:3,:3]@z-z for odo in odos]
        z_s_ = np.stack(z_s)
        ts_ = np.stack(ts)

        # motivation is large baseline and facing to the same obj
        #score = ((z_s_ + ts_)**2).sum(1)[:,0] / baselines_xy
        while True:
            invalid = ((np.absolute(epipole)<=epipole_th).sum(-1) > 0) #*\
                    #((np.absolute(back_epipole)<=epipole_th).sum(-1) > 0) 
            valid = ~invalid
            #valid = (np.absolute(epipole)>1).sum(-1) == 2
            valid = valid*(angles > cos_th) * (baselines > baseline_th) * (baselines < 1.)# * (baselines_xy > baseline_th)  
            #valid = (angles > cos_th) * (baselines > baseline_th) * (baselines < 1.)# * (baselines_xy > baseline_th)  




            if min_k is not None:
                if valid.sum() < min_k:
                    return None, None
                else:
                    break
            else:
                if valid.sum() < k:
                    return None, None
                    #cos_th-=.1
                    #epipole_th-=.1
                    baseline_th-=.01
                    if baseline_th < .1:
                    #cos_th -= .02
                        if relaxed:
                            if baseline_th < .02:
                                return None, None
                        else:
                            return None, None
                    continue
                else:
                    if valid.sum() < k:
                        k = valid.sum()
                    break
            

        # 2. compute parallax score
        z = np.array([[0,0,20]]).T #3x1
        z_s = [odo[:3,:3]@z-z for odo in odos]
        z_s_ = np.stack(z_s)
        ts_ = np.stack(ts)
        ts_[:,2,0] = 0 
        z_s_[:,2,0] = 0

        # option2 motivation is parallel
        #score = np.stack([((odo[:3,:3]@z+odo[:3,(3,)]-z)**2).sum() for odo in odos])

        # option1 motivation is large baseline and facing to the same obj
        score = 1/(baselines_xy+.000001) #((z_s_ + ts_)**2).sum(1)[:,0] / baselines_xy

        ''' 
        # min ||z-z^'|| / baselines
        z_s_ = np.stack([odo[:3,:3]@z+odo[:3,(3,)]-z for odo in odos])
        score = ((z_s_[:,:2])**2).sum(1)[:,0] / baselines
        '''

        score[~valid] = 100

        ids_ = np.argsort(score)
        ids = []
        # only keep valid ids
        for id in ids_:
            if valid[id]:
                ids.append(id)

        '''
        # use FPS to sample diversified samples
        # (1) compute the 2D coord that 0,0,2 located on each planar
        odos_back_ = [odos_back[id] for id in ids]
        z_s = [odo[:3,:3]@z for odo in odos_back_]
        xys = torch.as_tensor(np.stack(z_s))[:,:,0]# N_xys,3 
        # (2) FPS
        _, ids = pytorch3d.ops.sample_farthest_points(xys[None], K=k) 
        ids = ids[0,:].numpy().astype(int)    
        '''
        
        
        
        


        ids = ids[:k*10]
        #np.random.shuffle(ids)
        #print("[bold pink1]neibor ids",ids[:k])
        #print('[bold pin1]score', [score[id_] for id_ in ids[:k]])
        nb_odos = [odos[id_] for id_ in ids[:k]]
        if frames is not None:
            nb_ims = [frame for frame in [frames[id_] for id_ in ids[:k]]]
        #
        return nb_ims, np.stack(nb_odos) #k,4,4



    def scale_recovery(self, frame_depth, kps):
        ic('scale_recovery...')
        std_th = .05
        inlier_th = 20 

        H,W = frame_depth.shape
        #if (frame_depth>0).sum() < H//4*W//4:
        #    return None
        kps_3ds = kps.cpu().numpy() #N,3

        '''
        Twc = pose.matrix()
        '''
        Twc = np.eye(4)
        Tcw = np.linalg.inv(Twc) 
        R = Tcw[:3,:3]
        t = Tcw[:3,3:]
        kps_3ds = (R@kps_3ds.T + t).T # in cur frame's coord

        kps_proj_depth = kps_3ds[:,2]
        K = self.K.cpu().numpy() 
        kps_2ds = (K[:2,:2] @ (kps_3ds[:,:2]/kps_proj_depth[:,None]).T + K[:2,(2,)]).T.round().astype(int)
        valid = (kps_2ds[:,0] >=0) \
               *(kps_2ds[:,0] < W) \
               *(kps_2ds[:,1] >=0) \
               *(kps_2ds[:,1] < H) 
        kps_2ds = kps_2ds[valid,:]
        kps_proj_depth = kps_proj_depth[valid]
        kps_frame_depth = frame_depth[kps_2ds[:,1], kps_2ds[:,0]]

        valid_mask = (kps_proj_depth > .0001) * (kps_proj_depth < 10) * (kps_frame_depth > .0001) *\
                (~np.isnan(kps_proj_depth)) * (~np.isnan(kps_frame_depth)) *\
                (~np.isinf(kps_proj_depth)) * (~np.isinf(kps_frame_depth))


        # regression
        try:
            depth_ratio = kps_frame_depth[valid_mask] / kps_proj_depth[valid_mask]
            ic(depth_ratio.mean(), depth_ratio.std())
            if np.isnan(depth_ratio.mean()):
                print(kps_frame_depth[valid_mask], kps_proj_depth[valid_mask])
            else:
                print(depth_ratio.shape)
            # strategy 1: cancel if depth is bad
            # use ratio std
            assert depth_ratio.std()/depth_ratio.mean() < std_th, 'bad depth distribution < th'
            assert depth_ratio.shape[0]>10, "has too few depth_ratio"

            # fit
            ransac = linear_model.RANSACRegressor(
                        base_estimator=linear_model.LinearRegression(
                        fit_intercept=False),
                        min_samples=3,  # minimum number of min_samples
                        max_trials=1000, # maximum number of trials
                        stop_probability=.99,#0.99, # the probability that the algorithm produces a useful result
                        residual_threshold=0.001,  # inlier threshold value
                    )
            ransac.fit(
                depth_ratio.reshape(-1, 1),
                np.ones((depth_ratio.shape[0], 1))
            )
            #ic(ransac.score( depth_ratio.reshape(-1, 1),
            #    np.ones((depth_ratio.shape[0], 1))))

            scale = ransac.estimator_.coef_[0, 0]
            inliers = ransac.inlier_mask_
            ic(inliers.sum())
            assert inliers.sum()>inlier_th, "has too few inliers"
            frame_depth *= scale

            # filter out bad depth
            frame_depth[frame_depth < .01] = 0

        except Exception as e:
            print(e, 'bad depth estimation')
            return None

        # remove outliers
        if True:
            torch_depth = torch.as_tensor(frame_depth).to(self.device).float()
            dist_mask = torch_depth.view(-1) > .5
            pc_data = unproject_depth(torch_depth, *(self.intrinsics))
            pc_data = torch.cat([pc_data, torch.zeros((pc_data.size(0), pc_data.size(1), 1), device=pc_data.device)], dim=-1)
            pc_data = pc_data.reshape(-1,4)
            nan_mask = ~torch.isnan(pc_data[..., 0])
            with torch.cuda.device(self.device):
                valid_mask = remove_radius_outlier(pc_data, 16*4, 0.05) * dist_mask * nan_mask
                pc_data = pc_data[valid_mask]
                normal_data = estimate_normals(pc_data, 16*4, 0.1, [0.0, 0.0, 0.0])
                normal_valid_mask = ~torch.isnan(normal_data[..., 0])
                valid_mask[valid_mask.clone()] = normal_valid_mask
                valid_mask = valid_mask.unsqueeze(0)
            valid_mask = valid_mask.cpu().numpy().reshape(frame_depth.shape)
            frame_depth[~valid_mask] = 0
        

        return frame_depth
        
    def scale_recovery_w_poses(self, frame_depth, mvs_pose, gt_pose, scale_ratio_th=0.01, remove_outliers=True):
        if frame_depth is None:
            return None
        scales = []
        k = gt_pose.shape[0]
        for i in range(k):
            scale = np.sqrt((mvs_pose[i+1,:3]**2).sum() / (gt_pose[i,:3,3]**2).sum() )
            scales.append(scale)

        print("[bold pink1]scales:",scales)
        print("[bold pink1]scales std:", np.std(scales))
        if np.std(scales)/np.mean(scales) > scale_ratio_th or np.mean(scales) < .01:
            print("[bold pink1]bad scale, depth=None now")
            return None
        scale = np.mean(scales)
        frame_depth /= scale 
        # filter out bad depth
        frame_depth[frame_depth < .01] = 0

        if frame_depth.sum() < 100:
            print(frame_depth.sum())
            return None

        return frame_depth
    def remove_outliers(self, frame_depth):
        # remove outliers
        #if remove_outliers:

        if True:
            torch_depth = torch.as_tensor(frame_depth).to(self.device).float()
            dist_mask = torch_depth.view(-1) > .5
            pc_data = unproject_depth(torch_depth, *(self.intrinsics))
            pc_data = torch.cat([pc_data, torch.zeros((pc_data.size(0), pc_data.size(1), 1), device=pc_data.device)], dim=-1)
            pc_data = pc_data.reshape(-1,4)
            nan_mask = ~torch.isnan(pc_data[..., 0])
            with torch.cuda.device(self.device):
                # remove_radius_outlier cause error when radius is low
                #outlier_radius = 0.1 if np.median(frame_depth) > 5 else 0.05
                valid_mask = remove_radius_outlier(pc_data, 16*4, 0.05) * dist_mask * nan_mask
                pc_data = pc_data[valid_mask]
                normal_data = estimate_normals(pc_data, 16*4, 0.1, [0.0, 0.0, 0.0])
                normal_valid_mask = ~torch.isnan(normal_data[..., 0])
                valid_mask[valid_mask.clone()] = normal_valid_mask
                valid_mask = valid_mask.unsqueeze(0)
            valid_mask = valid_mask.cpu().numpy().reshape(frame_depth.shape)
            frame_depth[~valid_mask] = 0
        
        return frame_depth 




    def check_flow_triangulation(self, flow, certainty=None):
        '''
            
        '''
        st = time()
        N=flow.shape[1]
        #pc = depth23D(depth, self.intrinsics).reshape(-1,3) # HW,3 
        K = self.K.cpu().numpy()
        invK = np.linalg.inv(K)
        pc1 = invK[:2,:2]@self.px1 + invK[:2,(2,)] # 2,N
        pc1 = np.concatenate([pc1, np.ones((1,N))],axis=0) # 3,N
        


        # find the cooresponding pixel in neighbor camera
        px2 = self.px1+flow # 2,N
        pc2 = invK[:2,:2]@px2 + invK[:2,(2,)] # 2,N
        pc2 = np.concatenate([pc2, np.ones((1,N))],axis=0) # 3,N
        
        # essential matrix
        kRansacProb = 0.999
        try: 
            ransac_method = cv2.USAC_MSAC 
        except: 
            ransac_method = cv2.RANSAC  

        if certainty is not None:
            num = (certainty>0).sum()
            if num < 10:
                return certainty.reshape(-1)>0
            rand_idx = np.argsort(certainty.reshape(-1))[-num:]
            rand_idx = rand_idx[np.random.randint(0, num, size=2000)]
        else:
            rand_idx = np.random.randint(0, px2.shape[1], size=1000)
        E, mask_match = cv2.findEssentialMat(self.px1.T[rand_idx,:], px2.T[rand_idx,:], K, method=ransac_method, prob=kRansacProb)#, threshold=kRansacThresholdNormalized)                         
        ic(mask_match.sum())
        #print('solve Essential', time()-st) 
        _, R, t, mask = cv2.recoverPose(E, self.px1.T[rand_idx,:], px2.T[rand_idx,:], self.K.cpu().numpy(), mask=mask_match)
        T21 = np.eye(4)
        T21[:3,:3] = R
        T21[:3,(3,)] = t
        T12 = np.linalg.inv(T21)
        #print('recover pose', time()-st) 

        pc1 = torch.as_tensor(pc1).cuda()
        pc2 = torch.as_tensor(pc2).cuda()
        px2 = torch.as_tensor(px2).cuda()
        px1 = torch.as_tensor(self.px1).cuda()

        epipolar_line = torch.as_tensor(invK.T@E).cuda()@pc1#3,N
        den = (epipolar_line[:2,:]**2).sum(0)
        mask_epipolar_ = den != 0
        den[~mask_epipolar_] = 1 
        epipolar_dist = ((px2*epipolar_line[:2,:]).sum(0)+epipolar_line[2,:])**2 / den
        mask_epipolar_ *= epipolar_dist < (3.8 * (float(self.W) / 640) ** 2)

        # back check
        epipolar_line = (pc2.T@torch.as_tensor(E@invK).cuda()).T#3,N
        den = (epipolar_line[:2,:]**2).sum(0)
        mask_epipolar_ *= den != 0
        den[~mask_epipolar_] = 1 
        epipolar_dist = ((px1*epipolar_line[:2,:]).sum(0)+epipolar_line[2,:])**2 / den
        mask_epipolar_ *= epipolar_dist < (3.84 * (float(self.W) / 640) ** 2)


        '''
        epipolar_line2 = torch.as_tensor(invK.T@E).cuda()@pc1#3,N
        epipolar_line1 = (pc2.T@torch.as_tensor(E@invK).cuda()).T#3,N
        den = (epipolar_line2[:2,:]**2).sum(0) + (epipolar_line1[:2,:]**2).sum(0)
        mask_epipolar_ = den != 0
        den[~mask_epipolar_] = 1 
        sampson_dist = ((px2*epipolar_line2[:2,:]).sum(0)+epipolar_line2[2,:])**2 / den
        mask_epipolar_ *= sampson_dist < 1.92 #.92
        '''



        #print('static check', time()-st)
        '''
        point_4d_hom = cv2.triangulatePoints(K@np.eye(4)[:3,:], K@T21[:3,:], self.px1, px2)
        good_pts_mask = np.where(point_4d_hom[3]!= 0)[0]
        point_4d = point_4d_hom / point_4d_hom[3] 
        points_3d = point_4d[:3, :].T
        depth = points_3d[:,2].reshape(self.H,self.W)
        '''

    

        # cos
        rot = T12[:3,:3]#Quaternion(imaginary=cur_p[3:-1], real=cur_p[-1]).rotation_matrix
        #invRot = rot.T
        pc2_roted = torch.as_tensor(rot).cuda()@pc2
        cos_parallax_rays = (pc1*pc2_roted).sum(0) / torch.linalg.norm(pc1,axis=0) / torch.linalg.norm(pc2_roted,axis=0)
        mask_parallax = cos_parallax_rays < .9998 
        #print('parallax check', time()-st)

        # epipole
        # Fixed bug, it should be T12 
        ts = T12[:3,3].copy() #T21[:3,3] #k,3
        ts[:2] /= ts[(2,)]
        # should be z=1 pc1, but because pc1[2] is 1, donot need
        dist2epipole = pc1.T[:,:2] - torch.as_tensor(ts[None,:2]).cuda() #N,2
        mask_epipole = (dist2epipole**2).sum(-1) > 1./ 36

        # back epipole
        ts = T21[:3,3].copy() #k,3
        ts[:2] /= ts[(2,)]
        dist2epipole = pc2.T[:,:2] - torch.as_tensor(ts[None,:2]).cuda() #N,2
        mask_epipole = (dist2epipole**2).sum(-1) > 1./ 36


        mask= mask_parallax*mask_epipole*mask_epipolar_


        return mask.cpu().numpy()

        

    def depth_completion_cov(self, rgb, depth, monodepth, num_sample=200):
        '''
            input are cpu tensors with H,W,3 H,W H,W     
        '''

        # downsample
        H,W = monodepth.shape
        scale = depth / monodepth

        scale_ = F.interpolate(scale[None,None], (H//2,W//2), mode='nearest').squeeze()   
        rgb_ = F.interpolate(rgb.permute(2,0,1)[None]/255, (H//2,W//2), mode='bilinear',align_corners=False).squeeze() 


        # %32 == 0
        H_, W_ = (H//2)//32*32, (W//2)//32*32
        rgb_2 = rgb_[:,:H_,:W_]
        scale_2 = scale_[:H_,:W_]


        model_path = '/home/yijun/baselines/DepthCov/models/scannet.ckpt'
        size = rgb_2.shape[1:]
        #rgb_ = rgb_.permute(2,0,1)[None]/255 # 1,3,H,W
        #scale = depth_ / monodepth_

        coord_train = torch.nonzero(scale_2)
        perm = torch.randperm(coord_train.size(0))
        idx = perm[:num_sample] # 200
        coord_train = coord_train[idx]
        scale_train = scale_[coord_train[:,0], coord_train[:,1]]
        mean_scale = torch.mean(scale_train)

        model = NonstationaryGpModule.load_from_checkpoint(model_path, train_size=size)
        model.eval()
        model.to(self.device)
        # Run network
        gaussian_covs = model(rgb_2.cuda()[None])
        # Condition on sparse inputs
        coords_train_norm = normalize_coordinates(coord_train.view(1,-1,2), rgb_2.shape[-2:])
        pred_scales, pred_vars = model.condition(gaussian_covs, coords_train_norm.cuda(), scale_train.view(1,-1,1).cuda(), mean_scale.cuda(), size)

        #scale[scale==0] 
        pred_scale = pred_scales[-1].detach().squeeze().cpu()
        pred_var = pred_vars[-1].detach().squeeze().cpu()

        pred_scale_ = torch.zeros_like(scale_)
        pred_var_ = torch.zeros_like(scale_)
        pred_scale_[:H_,:W_] = pred_scale
        pred_var_[:H_,:W_] = pred_var
        '''
        mask_query = scale_2==0
        scale_2[mask_query] = pred_scale[mask_query]
        var_2 = torch.zeros_like(scale_2)
        var_1 = torch.zeros_like(scale_)
        var_2[mask_query] = pred_var[mask_query]
 
        scale_[:H_,:W_] = scale_2
        var_1[:H_,:W_] = var_2
        '''

        # upsample
        pred_scale = F.interpolate(pred_scale_[None,None], (H,W), mode='bilinear',align_corners=False).squeeze() 
        pred_var = F.interpolate(pred_var_[None,None], (H,W), mode='bilinear',align_corners=False).squeeze() 


        # completion
        mask_query = scale==0
        scale[mask_query] = pred_scale[mask_query]
        #cv2.imwrite('debug/scale.png', (scale.cpu().numpy()*1000).astype(np.uint16))
        var = torch.zeros_like(pred_var)
        var[mask_query] = pred_var[mask_query]
        
        return monodepth*scale, var


       








if __name__ == '__main__':
    from dataset import TUMRGBDDataset, ICLNUIMDataset, ReplicaRGBDDataset
    dataset = ReplicaRGBDDataset('/home/yijun/data/Replica/office0/results/') 
    traj = np.genfromtxt('/home/yijun/data/Replica/office0/traj.txt')

    H,W = dataset.rgb[0].shape[:2]
    fx, fy, cx, cy, scale = 600., 600., 599.5, 339.5, 6553.5
    K = np.eye(3,3)
    K[0,0] = fx
    K[1,1] = fy
    K[0,2] = cx
    K[1,2] = cy


    rgb0, depth0 = dataset[0]
    rgb1, depth1 = dataset[5]
    rgb2, depth2 = dataset[10]

    flower = Flower()
    s = 2
    flower.initialize(H//s,W//s,'cuda:0',(fx/s, fy/s, cx/s, cy/s))

    # 1. mvs with 2*nb+1 frames
    rgbs = [cv2.resize(dataset[i][0], [W//s,H//s], interpolation = cv2.INTER_CUBIC)   for i in range(0,21,5)]
    depth, cov, _ = flower.mvs(rgbs)
    #plt.imshow(cov)
    #plt.show()

    # 2. scale


    '''
    depth1 = depth1.astype(float)/ 6553.5
    depth = cv2.resize(depth, [W,H], interpolation = cv2.INTER_CUBIC)
    show_im = np.zeros((H,W*2))
    show_im[:,:W] = depth
    show_im[:,W:] = depth1
    plt.imshow(show_im)
    plt.show()
    '''


