import sys
import os
import numpy as np
import cv2
import time
import argparse 
import datetime


sys.path.append('../..')

import stereocam as sc
import handtrack as ht

if __name__ == '__main__':
    # Command line arguments 
    parser = argparse.ArgumentParser(description="Live tracking")
    parser.add_argument('-sp', '--save_path', default='../../data/tracking', type=str, help='Save path for tracking data storage.')
    parser.add_argument('-fz', '--frame_size', nargs="+", default=(672, 376), type=int, help='Frame size of stereo image.')
    parser.add_argument('-c', '--confidence_threshold', default=0.4, type=float, help='Confidence threshold for YOLO.')
    parser.add_argument('-fps', '--fps', default=10, type=int, help='Frames per second.')
    parser.add_argument('-sf', '--stereo_frame', default='left', type=str, help="Frame to use while tracking. Options: 'left', 'right'")
    
    save_path = str(save_path)
    frame_size = tuple(frame_size)
    confidence_threshold = float(confidence_threshold)
    fps = int(fps)
    stereo_frame = str(stereo_frame)

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
