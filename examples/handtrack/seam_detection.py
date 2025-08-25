import sys
import os
import numpy as np
import cv2
import time
import argparse 
import datetime
import copy


sys.path.append('../..')

import stereocam as sc
import handtrack as ht
import seamdet as sd

if __name__ == '__main__':
    # Command line arguments 
    parser = argparse.ArgumentParser(description="Live tracking")
    parser.add_argument('-sp', '--save_path', default='../../data/tracks', type=str, help='Save path for tracking data storage.')
    parser.add_argument('-fz', '--frame_size', nargs="+", default=(672, 376), type=int, help='Frame size of stereo image.')
    parser.add_argument('-c', '--confidence_threshold', default=0.4, type=float, help='Confidence threshold for YOLO.')
    parser.add_argument('-fps', '--fps', default=10, type=int, help='Frames per second.')
    parser.add_argument('-sf', '--stereo_frame', default='left', type=str, help="Frame to use while tracking. Options: 'left', 'right'")
    parser.add_argument('-cam', '--cam_param_path', default='../../data/calib/2025-08-15-19-07', type=str, help='Camera parameters')


    args = parser.parse_args()

    save_path = str(args.save_path)
    frame_size = tuple(args.frame_size)
    confidence_threshold = float(args.confidence_threshold)
    fps = int(args.fps)
    stereo_frame = str(args.stereo_frame)
    cam_params_path = os.path.join(args.cam_param_path, 'stereo_calib.npz')

    cam_data = sc.helpers.load_calibration_data(cam_params_path)

    Q = cam_data['Q']
    f = sc.calibration.get_focal_length(Q)
    baseline = sc.calibration.get_baseline(Q)

    timestamp = int(time.time())
    ct = datetime.datetime.now()
    date = ct.strftime("%Y-%m-%d-%H-%M")

    cam_index = sc.capture_images.detect_stereo_camera("zed")

    if not cam_index:
        print("Stereo camera not detected")
        sys.exit(1)

    model_path = ht.helpers.get_yolo_handpose_model('../../weights')

    session_save_path = os.path.join(save_path, str(date))

    os.makedirs(session_save_path, exist_ok=True)
    print(f"Created directory: {session_save_path}")

    live_det = ht.pose_track.YOLOHandPoseLiveRecord(cam=cam_index, 
                                                    fps=fps, 
                                                    filename=os.path.join(session_save_path, "tracks.csv"), 
                                                    model_path=model_path['n'], 
                                                    stereo_frame=stereo_frame, 
                                                    frame_size=frame_size,
                                                    confidence_threshold=confidence_threshold, 
                                                    **{'verbose': True})

    render_kw = {'font_color':(0, 0, 0),
                'label_font_color':(255, 255, 255),
                'label_font_scale':0.8,
                'label_font_thickness':2,
                'edge_color':(255, 255, 255), 
                'landmark_color':(0, 0, 255),
                'font':cv2.FONT_HERSHEY_SIMPLEX,
                'font_thickness':2,
                'box_color':(0, 0, 255),
                'box_thickness':5,
                'font_scale':0.5,
                'show_landmarks':True,
                'show_box':True,
                'show_label':True}

    live_det.stream(**render_kw)

    # Extracting first frame
    left_stereo, right_stereo = live_det.first_frame

    image_dir = os.path.join(session_save_path, "images")
    os.makedirs(image_dir, exist_ok=True)

    cv2.imwrite(os.path.join(image_dir, "left.png"), left_stereo)
    cv2.imwrite(os.path.join(image_dir, "right.png"), right_stereo)

    # ==============
    # Seam detection
    # ==============

    tracks_df, imgL, imgR = ht.helpers.load_tracks_data(session_save_path)

    # ---------------
    # Depth detection
    # ---------------

    grayL, grayR = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in [imgL, imgR]]

    # Stereo map generation
    stereoMapL, stereoMapR = sc.depth_estimation.stereo_map(cam_data, image_shape=grayL.shape[::-1])

    # Image rectification - hsv
    rectL, rectR = sc.depth_estimation.rectify_images(imgL, imgR, stereoMapL, stereoMapR)

    disparity, camera_projection, depth_map, left_cut = sc.depth_estimation.depth_maps(rectL, 
                                                                                        rectR, 
                                                                                        cam_data['Q'],
                                                                                        dispFactor=9, 
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

    cv2.imwrite(os.path.join(session_save_path, "disparity.png"), disparity)

    pcd = sc.depth_estimation.point_cloud(rectL, 
                                    depth_limits=(0,1), 
                                    camera_projection=camera_projection, 
                                    depth_map=depth_map,
                                    left_cut=None,
                                    visualize=False,
                                    save_path='.', 
                                    pcd_name=os.path.join(session_save_path, 'point_cloud.ply'),
                                    cloud_frame_size=0.05)
    
    # -----------
    # Track lines
    # -----------

    x = np.array(tracks_df['x']).astype(np.int32)
    y = np.array(tracks_df['y']).astype(np.int32)

    imgL_copy = copy.copy(imgL)

    for xi, yi in zip(x, y):
        cv2.circle(imgL_copy, (xi, yi), 2, (0,0,255), -1)

    # Saving unrectified image
    cv2.imwrite(os.path.join(session_save_path, "tracks_unrect.png"), imgL_copy)

    # Rectifying image and points
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^

    mtxL = cam_data['mtxL']
    distL = cam_data['distL']
    R1 = cam_data['R1']
    P1 = cam_data['P1']

    # Rectifying points
    x_rect, y_rect = sc.depth_estimation.rectify_points(x, y, mtxL, distL, R=R1, P=P1)

    rectL_copy = copy.copy(rectL)

    for xi, yi in zip(x_rect, y_rect):
        cv2.circle(rectL_copy, (xi, yi), 2, (0,0,255), -1)

    cv2.imwrite(os.path.join(session_save_path, "tracks_rect.png"), rectL_copy)

    # Projection point clouds
    # ^^^^^^^^^^^^^^^^^^^^^^^

    points3d = sc.depth_estimation.image_points_to_camera(x_rect, y_rect, camera_projection, max_depth=1, min_depth=0.4)
    sc.depth_estimation.visualise_points(pcd, points3d)

    # Seam area isolation
    # ^^^^^^^^^^^^^^^^^^^

    boxes = sd.seam.point_box(rectL, x_rect, y_rect, scaling_factor=10)

    roi_image = sd.seam.region_of_interest(rectL, boxes)

    cv2.imwrite(os.path.join(session_save_path, "roi_image.png"), roi_image)

    roi_patches = sd.seam.isolate_point_roi(rectL, boxes)

    patch_edges = sd.seam.roi_edges(roi_patches,
                        blur_n=0,
                        blur_kernel=(3,3)
                       )

    # Creating images of zero values
    img_zero = np.zeros(rectL.shape[0:2])

    patch_roi_img = sd.seam.patch_roi(img_zero, patch_edges)

    # Dilation

    dilation_kernel_size = (3,3)

    dilation_kernel = np.ones(dilation_kernel_size, np.uint8)

    img_dil = cv2.dilate(patch_roi_img, kernel=dilation_kernel, iterations=1)

    cv2.imwrite(os.path.join(session_save_path, "dilated_image.png"), img_dil)

    thinned = cv2.ximgproc.thinning(img_dil.astype(np.uint8))

    cv2.imwrite(os.path.join(session_save_path, "thinned.png"), thinned)

    # Extracting points
    # ^^^^^^^^^^^^^^^^^
    row_indices, col_indices = np.nonzero(thinned !=0)

    rectL_copy = copy.copy(rectL) 
    
    for xi, yi in zip(col_indices, row_indices):
        cv2.circle(rectL_copy, (xi, yi), 2, (0,0,255), -1)

    cv2.imwrite(os.path.join(session_save_path, "seam_rectL.png"), rectL_copy)

    points3d = sc.depth_estimation.image_points_to_camera(col_indices, row_indices, camera_projection, max_depth=1, min_depth=0.4)
    sc.depth_estimation.visualise_points(pcd, points3d)