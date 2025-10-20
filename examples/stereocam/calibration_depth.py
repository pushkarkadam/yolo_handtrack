import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import os 
import argparse
import time 
import yaml 
import datetime

sys.path.append('../..')

import stereocam as sc


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser(description="Live stereo calibration")

 
    parser.add_argument('-cam', '--cam_params_path', default='../../data/calib/2025-08-15-19-07', type=str, help='Camera parameters')
    # parser.add_argument('-cam', '--cam_name', default='zed', type=str , help='Camera name.')
    # parser.add_argument('-cz', '--chessboard_size', nargs="+", default=(8,6), type=int , help='Grids inside the chessboard.')
    # parser.add_argument('-vsf', '--view_scaling_factor', default=1, type=int , help='Scaling factor while vieweing live calibration.')
    parser.add_argument('-fs', '--frame_size', nargs="+", default=(672, 376), type=int , help='Resolution of the image.')
    # parser.add_argument('-sp', '--save_path', default='../../data/calib', type=str , help='Path to storing calibration data.')
    # parser.add_argument('-sq', '--square_size', default=0.039, type=float, help='Size of each square of the chessboard.')
    # parser.add_argument('-lim', '--image_limit', default=30, type=int , help='Maximum number of images stored.')
    # parser.add_argument('-sr', '--save_rendered', default='rendered', type=str , help='Stores the rendered image.')
    parser.add_argument('-df', '--dispFactor', default=12, type=int, help='Disparity factor.')
    parser.add_argument('-bs', '--blockSize', default=5, type=int, help='Block size for stereo matching.')
    parser.add_argument('-md', '--minDisparity', default=0, type=int, help='Minimum disparity.')
    parser.add_argument('-ddiff', '--disp12MaxDiff', default=-1, type=int, help='disp12MaxDiff.')
    parser.add_argument('-pfc', '--preFilterCap', default=30, type=int , help='preFilterCap.')
    parser.add_argument('-ur', '--uniquenessRatio', default=0, type=int , help='uniquenessRatio.')
    parser.add_argument('-sw', '--speckleWindowSize', default=50, type=int, help='speckleWindowSize.') # range 50-200
    parser.add_argument('-specr', '--speckleRange', default=2, type=int , help='Speckle range.') # range 1 or 2
    parser.add_argument('-mode', '--mode' , default=0, type=int, help='SGBM mode.')
    parser.add_argument('-ity', '--image_type', default='bgr', type=str, help='Image conversion.')

    args = parser.parse_args()

    calib_save_path = str(args.cam_params_path)
    # cam_name = str(args.cam_name)
    # chessboard_size = tuple(args.chessboard_size)
    # view_scaling_factor = int(args.view_scaling_factor)
    frame_size = tuple(args.frame_size)
    # save_path = str(args.save_path)
    # square_size = float(args.square_size)
    # image_limit = int(args.image_limit)
    # save_rendered = str(args.save_rendered)
    dispFactor = int(args.dispFactor)
    blockSize = int(args.blockSize)
    minDisparity = int(args.minDisparity)
    disp12MaxDiff = int(args.disp12MaxDiff)
    preFilterCap = int(args.preFilterCap)
    uniquenessRatio = int(args.uniquenessRatio)
    speckleWindowSize = int(args.uniquenessRatio) # range 50-200
    speckleRange = int(args.speckleRange) # range 1 or 2
    mode = int(args.mode)
    image_type = str(args.image_type)

    cam_params_path = os.path.join(calib_save_path, 'stereo_calib.npz')

    # cam_index = sc.capture_images.detect_stereo_camera(cam_name)


    # error, calib_save_path = sc.calibration.stereo_live_calibration(
    #                             cam=cam_index,
    #                             chessboard_size=chessboard_size,
    #                             view_scaling_factor=view_scaling_factor,
    #                             frame_size=frame_size,
    #                             save_path=save_path,
    #                             square_size=square_size,
    #                             image_limit=image_limit,
    #                             save_rendered=save_rendered
    #                             )

    # camera data
    # cam_data = np.load(os.path.join(calib_save_path, 'stereo_calib.npz'))
    # cam_data = sc.helpers.load_calibration_data(os.path.join(calib_file_path, 'stereo_calib.npz'))

    cam_data = sc.helpers.load_calibration_data(cam_params_path)

    Q = cam_data['Q']
    focal_length = sc.calibration.get_focal_length(Q)
    baseline = sc.calibration.get_baseline(Q)

    # Q = cam_data['Q']

    # baseline = 1/Q[-1, 2]
        
    # focal_length = Q[2, -1]

    # Image
    img_num = 0

    imageL, imageR = [cv2.imread(os.path.join(calib_save_path, f'stereo_{cam}' ,f'{cam}_img{img_num}.png')) for cam in ['left', 'right']]
    height, width, _ = imageL.shape

    # Computing stereo map
    stereoMapL, stereoMapR = sc.depth_estimation.stereo_map(cam_data, image_shape=(width, height))

    # Rectifying the images
    rectL, rectR = sc.depth_estimation.rectify_images(imageL, imageR, stereoMapL, stereoMapR)

    # Calculating depth maps

    disparity, camera_projection, depth_map, left_cut = sc.depth_estimation.depth_maps(rectL, 
                                                                                    rectR, 
                                                                                    Q,
                                                                                    dispFactor=dispFactor, 
                                                                                    blockSize=blockSize,
                                                                                    minDisparity=minDisparity,
                                                                                    disp12MaxDiff=disp12MaxDiff,
                                                                                    preFilterCap=preFilterCap,
                                                                                    uniquenessRatio=uniquenessRatio,
                                                                                    speckleWindowSize=speckleWindowSize, # range 50-200
                                                                                    speckleRange=speckleRange, # range 1 or 2
                                                                                    mode = mode,
                                                                                    image_type=image_type
                                                                                )

    cv2.imwrite(os.path.join(calib_save_path, "disparity.png"), disparity)

    print(f"Estimated center depth from depth_map: {depth_map[int(height/2), int(width/2)]} m")

    # Saving depth maps
    np.savez(os.path.join(calib_save_path, "depth_data.npz"), disparity, camera_projection, depth_map, left_cut)

    timestamp = int(time.time())

    # Storing calibration data
    # calib_params = {
    #     "calib_error": float(error),
    #     "baseline": float(baseline),
    #     "focal_length": float(focal_length),
    #     "image_size": {"width": width, "height": height},
    #     "calib_hash": f"calib_{timestamp}",
    #     "chessboard_size": {"h": chessboard_size[0], "v": chessboard_size[1]},
    #     "square_size": float(square_size),
    # }

    # with open(os.path.join(calib_save_path, f'calibration_{timestamp}.yaml'), 'w') as file:
    #     yaml.dump(calib_params, file, default_flow_style=False)

    # Developing point cloud
    rgb_rectL = cv2.cvtColor(rectL, cv2.COLOR_BGR2RGB)

    pcd = sc.depth_estimation.point_cloud(rectL, 
                    depth_limits=(0,1), 
                    camera_projection=camera_projection, 
                    depth_map=depth_map,
                    left_cut=left_cut, 
                    cloud_frame_size=0.05,
                    save_path='.', 
                    pcd_name=os.path.join(calib_save_path, f'pointcloud_{timestamp}.ply'))