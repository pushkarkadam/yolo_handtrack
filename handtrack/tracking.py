import numpy as np 
import cv2
import sys 
import yaml

sys.path.append("../")

import handtrack as ht
import stereocam as sc


def param_loader(filename):
    """Loades yaml file"""
    with open(filename) as stream:
        try:
            print(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)

    return stream

def point_depth_estimation(image0_pair,
                           left_image,
                           cam_params,
                           disp_params
                          ):
    """Estimates 3d point"""
    # TODO: Create the depth point estimation function


def load_tracks(stereo_dataloader, model_path, cam_data_path, confidence_threshold=0.2, KP=8, verbose=False):
    """Extracts tracks from the data.

    Parameters
    ----------
    stereo_dataloader: torch.utils.data.dataloader.DataLoader
        Dataloader from PyTorch that loads batch of images.
    model_path: str
        Path to the trained model.
    cam_data_path: str
        Path to stereo calibration data. 
    confidence_threshold: float, default ``0.2``
        Threshold to consider while detecting.
    KP: int, default ``8``
        Keypoint index to track. ``8`` indicates the tip of the index finger.
    verbose: bool, default ``False``
        If ``True``, then prints out the detection results in the console.

    Returns
    -------
    detection_tracks: list
        A list of tuples of the coordinates of the key point being tracked.
    
    """
    # stores the detection coordinates
    detection_track = []

    for left_images, right_images, image_names in tqdm(stereo_dataloader):
        left_frames = frames_torch_to_numpy(left_images)
        right_frames = frames_torch_to_numpy(right_images)

        # Rectification
        left_frames_rect = []
        right_frames_rect = []

        for left_frame, right_frame in zip(left_frames, right_frames):
            # rectifying images before detection
            left_frame_rect, right_frame_rect = ht.helpers.rectify_stereo_images(imageL=left_frame, imageR=right_frame, cam_data_path=cam_data_path)
            
            left_frames_rect.append(left_frame_rect)
            right_frames_rect.append(right_frame_rect)

        # YOLO-handpose detection
        render = YOLOHandPose(frames=left_frames_rect, model_path=model_path, confidence_threshold=confidence_threshold)

        # Processing all the frames
        render.process(verbose=verbose)
        xyn = render.xyn

        # List to stroe the tracks
        track_coords = []

        for coord in xyn:
            keypoints = coord[0]
            if keypoints:
                track_coords.append(keypoints[KP])
            else:
                track_coords.append(None)

        for track_coord in track_coords:
            detection_track.append(track_coord)

    return detection_track

def generate_depth_map(stereo_dataloader, cam_data_path, disparity_config_file):
    """Generates depth map using first frame of the video.
    
    Parameters
    ----------
    stereo_dataloader: torch.utils.data.dataloader.DataLoader
        Dataloader from PyTorch that loads batch of images.
    cam_data_path: str
        Path to stereo calibration data. 
    disparity_config_file: str
        Path to the disparity config file. The file type must be a YAML.

    Returns
    -------
    dict
        A dictionary with keys: disparity, camera_projection, depth_map, left_cut, pcd
        
    """
    if isinstance(cam_data_path, str):
        calib_data = np.load(cam_data_path, allow_pickle=True)
    else:
        calib_data = cam_data_path

    # Iterating only once on the dataloader
    left_images, right_images, image_names = next(iter(stereo_dataloader))

    # Extracting the first image from the dataloader
    imageL = left_images[0].permute(1,2,0).numpy()
    imageR = right_images[0].permute(1,2,0).numpy()

    # rectifying left and right frames
    left_frame_rect, right_frame_rect = ht.helpers.rectify_stereo_images(imageL=imageL, imageR=imageR, cam_data_path=cam_data_path)

    # converting to gray images
    grayL, grayR = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in [imageL, imageR]]

    # Converting to HSV colorspace
    imgL_hsv, imgR_hsv = sc.helpers.colorspace_transform(imageL, imageR)
    
    # filtering the images
    imgL_fil, imgR_fil = sc.helpers.clahe_filter(imgL_hsv, imgR_hsv)
    
    # Stereo map generation
    stereoMapL, stereoMapR = sc.depth_estimation.stereo_map(calib_data, image_shape=grayL.shape[::-1])
    
    # Image rectification - hsv
    rectL, rectR = sc.depth_estimation.rectify_images(imgL_fil, imgR_fil, stereoMapL, stereoMapR)
    
    # image rectification - bgr
    bgrRectL, bgrRectR = sc.depth_estimation.rectify_images(imageL, imageR, stereoMapL, stereoMapR)

    # Getting depth estimation disparity params from config file
    disp_params = sc.helpers.load_stereo_params(disparity_config_file)
    
    # Depth maps generation
    disparity, camera_projection, depth_map, left_cut = sc.depth_estimation.depth_maps(imageL=rectL, 
                                                                                       imageR=rectR, 
                                                                                       Q=calib_data['Q'],
                                                                                       **disp_params
                                                                                      )

    pcd = sc.depth_estimation.point_cloud(bgrRectL, 
                                         depth_limits=(0, 0.5), 
                                         camera_projection=camera_projection, 
                                         depth_map=depth_map, 
                                         left_cut=left_cut, 
                                         image_type='bgr', 
                                         visualize=False)

    depth_data = {'disparity': disparity,
                  'camera_projection': camera_projection,
                  'depth_map': depth_map,
                  'left_cut': left_cut,
                  'pcd': pcd
                 }
    
    return depth_data