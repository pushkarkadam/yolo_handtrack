# from ultralytics import YOLO
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
# from torchvision import transforms
# from tqdm import tqdm
import open3d as o3d
# import torch
import time 
import random
import csv


sys.path.append('../..')

import stereocam as sc
import handtrack as ht

if __name__ == '__main__':

    save_path = '../../data/tracking'
    frame_size = (672, 376)
    confidence_threshold = 0.4
    fps = 10
    stereo_frame = 'left'

    timestamp = int(time.time())

    cam_index = sc.capture_images.detect_stereo_camera("zed")

    if not cam_index:
        print("Stereo camera not detected")
        sys.exit(1)

    model_path = ht.helpers.get_yolo_handpose_model('../../weights')

    session_save_path = os.path.join(save_path, str(timestamp))

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

    image_dir = os.path.join(session_save_path, "images")
    os.makedirs(image_dir, exist_ok=True)

    cv2.imwrite(os.path.join(image_dir, "left.png"), left_stereo)
    cv2.imwrite(os.path.join(image_dir, "right.png"), right_stereo)
