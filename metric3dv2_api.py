import os, sys
import argparse
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import pdb

try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction
current_path = os.path.dirname(os.path.abspath(__file__))
metric3d_dir=current_path+'/thirdparty/Metric3D/'
sys.path.append(metric3d_dir)
from mono.model.monodepth_model import get_configured_monodepth_model
from mono.utils.do_test import transform_test_data_scalecano



def get_mono_api():
    config_large = Config.fromfile(metric3d_dir+'/mono/configs/HourglassDecoder/vit.raft5.large.py')

    model_large = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
    model_large.cuda().eval()


    def fn(image, inference_size=(340,600), intrinsic=[300.,300.,599.5/2, 339.5/2]):
        '''
            image: H,W,3
        '''
        ################################# Comment the early release huggingface version of v2
        ori_size = image.shape[:2]
        if inference_size is not None:
            img = cv2.resize(image, inference_size[::-1])
        else:
            img = image
        if intrinsic is None:
            intrinsic = [1000.0, 1000.0, img.shape[1]/2, img.shape[0]/2]
        rgb_input, cam_models_stacks, pad, label_scale_factor = transform_test_data_scalecano(img, intrinsic, config_large.data_basic)

        with torch.no_grad():
            #pred_depth, pred_depth_scale, scale, output = get_prediction(
            '''
            pred_depth, output = get_prediction(

                    model = model_large,
                    input = rgb_input[None].cuda(),
                    cam_model = cam_models_stacks,
                    pad_info = pad,
                    scale_info = label_scale_factor,
                    gt_depth = None,
                    normalize_scale = config_large.data_basic.depth_range[1],
                    ori_shape=[img.shape[0], img.shape[1]],
                )
            '''
            pred_depth, confidence, output = model_large.inference({'input': rgb_input[None]})

            
            pred_normal = output['prediction_normal'][:, :3, :, :] #output['normal_out_list'][0][:, :3, :, :] 
            H, W = pred_normal.shape[2:]
            pred_normal = pred_normal[:, :, pad[0]:H-pad[1], pad[2]:W-pad[3]]


        pred_depth = pred_depth.squeeze()#.cpu().numpy()
        pred_depth[pred_depth<0] = 0
        #pred_color = gray_to_colormap(pred_depth)
        pred_normal = pred_normal.squeeze()
                #pred_color_normal = vis_surface_normal(pred_normal)
        

        if inference_size is not None:
            pred_depth = F.interpolate(pred_depth[None,None,:,:], size=ori_size, mode='bilinear').squeeze()
            pred_normal = F.interpolate(pred_normal[None,:,:,:], size=ori_size, mode='bilinear').squeeze()

        if pred_normal.size(0) == 3:
            pred_normal = pred_normal.permute(1,2,0)

        return pred_depth.cpu().numpy(), pred_normal.cpu().numpy()
        '''
        inp = torch.tensor(image).float()/255
        inp = inp.permute(2,0,1)[None].cuda() # 1x3xHxW
        pred_depth, confidence, output_dict = model_large.inference({'input': inp})
        pred_normal = output_dict['prediction_normal'][:, :3, :, :]
        return pred_depth.squeeze().cpu().numpy(), pred_normal[0].permute(1,2,0).cpu().numpy()
        '''

    return fn




if __name__ == '__main__':

    monodepth = get_mono_api()
    img1 = cv2.imread(sys.argv[1])
    
    depth, normal = monodepth(img1)
    pdb.set_trace()
    
    plt.imshow(depth)
    plt.show()


