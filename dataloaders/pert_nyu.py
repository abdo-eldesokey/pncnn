import numpy as np

from .nyu_camera_parameters import *


def rgb_plane2rgb_world(depth_image):
    # Transforms depth image from RGB plane to RGB camera coordinate system


    rows, cols = depth_image.shape
    
    x = np.arange(0, cols, 1) 
    y = np.arange(0, rows, 1) 
    x_plane, y_plane = np.meshgrid(x, y)
    
    X_world = (x_plane-cx_rgb) * depth_image / fx_rgb 
    Y_world = (y_plane-cy_rgb) * depth_image / fy_rgb 
    Z_world = depth_image 
    
    points3d = np.array([X_world.flatten('F'), Y_world.flatten('F'), Z_world.flatten('F')]).transpose();
      
    return points3d


def rgb_world2depth_world(points3d_rgb):
# Transforms from RGB world coordinate to Depth world coordinate

    T = np.array([[t_x, t_z, t_y]]).transpose()
    #points3d_d = np.linalg.inv(R) @ (points3d_rgb.transpose() - T @ np.ones((1, points3d_rgb.shape[0])));    
    points3d_d = (points3d_rgb.transpose() - T @ np.ones((1, points3d_rgb.shape[0])));    

    return points3d_d.transpose() 


def depth_world2rgb_world_shifted(points3d, shift_x):
    
    shift = np.array([[shift_x/100, 0, 0]]).transpose()
    T = np.array([[t_x, t_z, t_y]]).transpose()
  
    #points3d = (R @ points3d.transpose()) + ((T+shift) @ np.ones((1, points3d.shape[0])));
    points3d = points3d.transpose() + (T+shift) @ np.ones((1, points3d.shape[0]));
    
    return points3d.transpose()


def depth_world2rgb_world_rotated(points3d, rotation_z):

    T = np.array([[t_x, t_z, t_y]]).transpose()

    orig_R = R
    orig_angle = np.rad2deg(np.arccos(orig_R[0,0]))
    new_angle = np.deg2rad(orig_angle + rotation_z)
    new_R = np.array([[np.cos(new_angle), -np.sin(new_angle), 0],
                      [np.sin(new_angle), np.cos(new_angle), 0],
                      [0,0,-1]])

    points3d = (new_R @ points3d.transpose()) + ((T) @ np.ones((1, points3d.shape[0])));

    return points3d.transpose()


def rgb_world2rgb_plane(points3d):

  X_world = points3d[:,0]
  Y_world = points3d[:,1]
  Z_world = points3d[:,2]
  
  mask = Z_world==0
  Z_world[mask] = -1
  
  X_plane = (X_world * fx_rgb / Z_world) + cx_rgb
  Y_plane = (Y_world * fy_rgb / Z_world) + cy_rgb
  
  X_plane[mask]=0
  Y_plane[mask]=0
  
  return X_plane, Y_plane 


class ShiftDepth(object):

    def __init__(self, shift=0, rotate=0):
        self.shift = shift
        self.rotate = rotate

    def __call__(self, input_depth):
        points3d_rgb = rgb_plane2rgb_world(input_depth)
        points3d_d = rgb_world2depth_world(points3d_rgb)
        if self.shift != 0:
            points3d_d = depth_world2rgb_world_shifted(points3d_d, self.shift)

        if self.rotate != 0:
            points3d_d = depth_world2rgb_world_rotated(points3d_d, self.rotate)

        xProj, yProj = rgb_world2rgb_plane(points3d_d)
        
        # Finally, project back onto the RGB plane.
        H, W = input_depth.shape

        assert(H==480)
        assert(W==640)
        
        xProj = np.round(xProj).astype(np.int)
        yProj = np.round(yProj).astype(np.int)

        goodInds = np.argwhere((xProj>=0) & (xProj<W) &(yProj>=0) & (yProj<H))

        input_depth_fl = input_depth.flatten('F')
        order = np.argsort(-input_depth_fl[goodInds], axis=0)
        depthSorted = -np.sort(-input_depth_fl[goodInds], axis=0)

        #[r,c] = np.unravel_index(goodInds, input_depth.shape, order='F')

        depthOut = np.zeros(H*W)
        coord_x = xProj[goodInds[order]].squeeze()
        coord_y = yProj[goodInds[order]].squeeze()
        idxss = np.expand_dims(np.ravel_multi_index((coord_y, coord_x), (480,640), order='F'), -1)
        depthOut[idxss] = depthSorted
        depthOut = depthOut.reshape(input_depth.shape, order='F')
        
        #depthOut = np.zeros(input_depth.shape)
        #for ii in range(len(order)):
        #    depthOut[yProj[goodInds[order[ii]]], xProj[goodInds[order[ii]]]] = depthSorted[ii]

        depthOut[depthOut > maxDepth] = maxDepth
        depthOut[depthOut < 0] = 0
        #depthOut[np.isnan(depthOut)] = 0
        return depthOut
            

import h5py
def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])
    return rgb, depth


def shift_image(path, shift):
    rgb, depth = h5_loader(path)
    shift_depth_transform = ShiftDepth(shift=shift)
    shifted_depth = shift_depth_transform(depth)
    
    # Fix the wrong colormap issue
    if shift==0:
        shifted_depth = np.pad(shifted_depth, (1,0), 'constant')
    
    
    import matplotlib.pylab as plt
    import os 
    fname = os.path.basename(path)
    plt.imsave(fname[:5] + '_shift_' + str(shift) + '.jpg', shifted_depth)
        
if __name__ == '__main__':
    paths = ['/media/abdel62/Data/nyu-depth-v2/nyudepthv2/val/official/00396.h5']
    
    for path in paths:
        shift_image(path,10)
    
    
    

        
    
    
    
    
    
    
