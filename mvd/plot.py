import os, sys
import torch
import open3d as o3d
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_path))
from ext import unproject_depth

def depth23D(torch_depth, intrinsic):
    #intrinsic = [intrinsic_matrix[0,0], intrinsic_matrix[1,1], intrinsic_matrix[0,2], intrinsic_matrix[1,2]]
    pc_data = unproject_depth(torch_depth, *(intrinsic))
    return pc_data
 
def plot_depth_as_pc(depth, intrinsic_matrix):
    intrinsic = [intrinsic_matrix[0,0], intrinsic_matrix[1,1], intrinsic_matrix[0,2], intrinsic_matrix[1,2]]
    torch_depth = torch.from_numpy(depth).cuda().float()

    pc_data = unproject_depth(torch_depth, *(intrinsic))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_data.cpu().numpy().reshape(-1,3))
    o3d.io.write_point_cloud('debug.pcd', pcd) 
    #o3d.visualization.draw_geometries([pcd])

