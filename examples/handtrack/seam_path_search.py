import sys
import os
import numpy as np
import cv2
import time
import argparse 
import datetime
import copy
import pandas as pd
import pickle


sys.path.append('../..')

import stereocam as sc
import handtrack as ht
import seamdet as sd

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser(description="Search Path")
    parser.add_argument('-p', '--path')
    parser.add_argument('-cam', '--cam_params_path', default='../../data/calib/2025-08-26-16-47', type=str, help='Camera parameters')
    parser.add_arguments('-hp', '--hand_eye_path', default='../../data/hand_eye_calibration/1764800285/', type=str, help='Hand Eye calibration parameter path')

    args = parser.parse_args()

    # Extracting command line variables
    track_path = str(args.path)
    cam_params_path = os.path.join(args.cam_params_path, 'stereo_calib.npz')
    hand_eye_calibration_path = str(args.hand_eye_path)

    # Extracting camera data
    cam_data = sc.helpers.load_calibration_data(cam_params_path)

    Q = cam_data['Q']
    f = sc.calibration.get_focal_length(Q)
    baseline = sc.calibration.get_baseline(Q)

    # Loading camera projection
    with open(os.path.join(track_path, 'camera_projection.npy'), 'rb') as f:
        camera_projection = np.load(f)

    I = cv2.imread(os.path.join(track_path, 'thinned.png'), 0)

    I = np.where(I>=1, 1, 0)

    coords = sd.path.get_path_coords(I)

    end_nodes = sd.path.get_end_nodes(I, coords)

    rect_tracks_df = pd.read_csv(os.path.join(track_path, "rect_tracks.csv"))

    tracking_ends = sd.path.get_tracking_endpoints(rect_tracks_df)

    start, goal = sd.path.true_end_nodes(end_nodes, tracking_ends)

    start_time = time.time()

    seam_path = sd.path.search_path(start, goal, I)

    print(f'Time elapsed: {(time.time() - start_time):.4f}s')

    path_ancestory = sd.path.get_path_ancestry(seam_path, start, goal)

    seam_path_file = os.path.join(track_path, "seam_path_coords.pkl")

    # seam path
    xl, yl = zip(*path_ancestory)

    seam_path_dict = {'x': xl, 'y': yl}
    
    with open(seam_path_file, 'wb') as f:
        pickle.dump(seam_path_dict, f)

    # Convert image points to camera coordinates
    camera_coords = sc.depth_estimation.image_points_to_camera(np.array(xl), np.array(yl), camera_projection)

    camera_coords_df = sc.depth_estimation.get_camera_coords_df(camera_coords, save_path=os.path.join(track_path, 'seam_path_camera_coords.csv'))

    # Robot coordinates

    rTc_path = os.path.join(hand_eye_calibration_path, 'rTc.npz')

    robot_coords_df = sd.path.get_robot_coords(hand_eye_calibration_path, camera_coords_df, save_path=os.path.join(track_path, "seam_path_robot_coords.csv"))