import cv2
import numpy as np
from flower import Flower
#from plot import depth23D
import matplotlib.pyplot as plt
import pdb

if __name__ == '__main__':

    # prepare data
    rgb_paths = ['./data/frame%06d.jpg'%i for i in range(0,100,20)] 
    rgbs = [cv2.cvtColor(cv2.imread(rgb_path,-1),cv2.COLOR_BGR2RGB) for rgb_path in rgb_paths]
    H,W,_ = rgbs[0].shape

    fx, fy, cx, cy, scale = 600., 600., 599.5, 339.5, 6553.5
    calib = [fx,fy,cx,cy]
    s = 2
    calib = [item/2 for item in calib]
    H = H//2
    W = W//2
    rgbs = [cv2.resize(rgb, [W,H], interpolation = cv2.INTER_CUBIC) for rgb in rgbs]
    '''
    K = np.eye(3,3)
    K[0,0] = fx
    K[1,1] = fy
    K[0,2] = cx
    K[1,2] = cy
    '''

    # rgbs = [ref]+[src1,src2,...]
    # mvd for the first frame in rgbs
    flower = Flower()
    flower.initialize(H,W,'cuda:0', calib)
    depth, intrinsics, poses, mono_info = flower.mvs(rgbs)
    plt.imshow(depth) 
    plt.show()

