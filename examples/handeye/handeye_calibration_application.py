import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
import time 
import copy
from datetime import datetime
import pandas as pd
import yaml


sys.path.append('../..')

import stereocam as sc
import handeye


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser(description="Robot Hand to Eye Calibration")
    parser.add_argument('-p', '--path', default='/home/robot1/Documents/hand_eye_calibration', type=str, help="Calibration path to store the calibration result.")
    parser.add_argument('-cam', '--cam_params_path', default='../../data/calib/2025-08-26-16-47', type=str, help='Camera parameters')

    args = parser.parse_args()

    # Extracting command line variables
    root_path = str(args.path)
    cam_params_path = os.path.join(args.cam_params_path, 'stereo_calib.npz')

    # ==========
    # Save paths
    # ==========

    timestamp = int(time.time())

    save_path = os.path.join(root_path, str(timestamp))

    os.makedirs(os.path.join(save_path), exist_ok=True)

    # =====================
    # Stereo camera capture
    # =====================

    cam_num = sc.capture_images.detect_stereo_camera(camera_name="zed")

    imgL, imgR = sc.capture_images.capture_stereo_image(cam_num, image_resolution=(672, 376))

    cv2.imwrite(os.path.join(save_path, "left_img.png"), imgL)

    # ========================
    # Stereo camera parameters
    # ========================

    cam_data = sc.helpers.load_calibration_data(camera_params_path)

    f = sc.calibration.get_focal_length(cam_data['Q'])
    baseline = sc.calibration.get_baseline(cam_data['Q'])

    # ==============
    # Loading images
    # ==============

    grayL, grayR = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in [imgL, imgR]]

    # Stereo map generation
    stereoMapL, stereoMapR = sc.depth_estimation.stereo_map(cam_data, image_shape=grayL.shape[::-1])

    # Image rectification
    rectL, rectR = sc.depth_estimation.rectify_images(imgL, imgR, stereoMapL, stereoMapR)


    disparity, camera_projection, depth_map, left_cut = sc.depth_estimation.depth_maps(rectL, 
                                                                                    rectR, 
                                                                                    cam_data['Q'],
                                                                                    dispFactor=12, 
                                                                                    blockSize=5,
                                                                                    minDisparity=0,
                                                                                    disp12MaxDiff=-1,
                                                                                    preFilterCap=30,
                                                                                    uniquenessRatio=1,
                                                                                    speckleWindowSize=100, # range 50-200
                                                                                    speckleRange=1, # range 1 or 2
                                                                                    mode = 0,
                                                                                    image_type='rgb',
                                                                                    remove_stereo_blank=False
                                                                                    )

    cv2.imwrite(os.path.join(save_path,"rectL.png"), rectL)
    cv2.imwrite(os.path.join(save_path, "disparity.png"), disparity)

    # Loading camera projection
    with open(os.path.join(save_path, 'camera_projection.npy'), 'rb') as f:
        camera_projection = np.load(f)

    pcd = sc.depth_estimation.point_cloud(rectL, 
                                    depth_limits=(0,1), 
                                    camera_projection=camera_projection, 
                                    depth_map=depth_map,
                                    left_cut=None, 
                                    save_path=save_path, 
                                    pcd_name=os.path.join(save_path, "point_cloud.ply"),
                                    cloud_frame_size=0.05)

    imgpoints, rect_img_render = handeye.get_image_points(rectL)

    cv2.imwrite(os.path.join(save_path, "rect_img_render.png"), rect_img_render)

    np.savez(os.path.join(save_path, "imgpoints.npz"), imgpoints)

    pcd = sc.depth_estimation.point_cloud(rect_img_render, 
                                    depth_limits=(0,1), 
                                    camera_projection=camera_projection, 
                                    depth_map=depth_map,
                                    left_cut=None, 
                                    save_path=save_path, 
                                    pcd_name="render.ply",
                                    cloud_frame_size=0.05)

    # ====================
    # Hand Eye Calibration
    # ====================

    # ---------------
    # World to camera
    # ---------------

    cTw = handeye.T_world_to_camera(imgpoints, cam_data['Q'], disparity)

    handeye.save_transformation(cTw, file_name="cTw", save_path=save_path)

    # --------------
    # World to robot
    # --------------

    p1, p2, p3 = handeye.read_robot_calibration_points(save_path)

    # ---------------
    # Camera to robot
    # ---------------

    rTc = handeye.T_camera_to_robot(cTw, rTw)

    handeye.save_transformation(rTc, file_name="rTc", save_path=save_path)

    # ==========
    # Experiment
    # ==========

    robot_coords_list = [handeye.robot_projection_evaluation(i ,imgpoints, cam_data['Q'], disparity, rTc) for i in [0, 40, 7]]

    cam_coords, robot_coords = zip(*robot_coords_list)

    xrms, yrms, zrms = handeye.reprojection_error((p1,p2,p3), robot_coords, save_path=save_path)

    print(f'x: {xrms}')
    print(f'y: {yrms}')
    print(f'z: {zrms}')









