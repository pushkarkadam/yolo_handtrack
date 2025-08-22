import cv2
import argparse
import time
import os
import sys 

sys.path.append('../')
from stereocam import *


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Stereo Camera Video Capture')
    parser.add_argument('-o','--output_path', default='../images', type=str, help='Path to store the videos')
    parser.add_argument('-c','--camera', default=4, type=int, help='Camera number')
    parser.add_argument('-width', "--width", default=3840, help="Width of the image resolution. Defaults to 3840.")
    parser.add_argument('-height', "--height", default=1080, help="Height of the image resolution. Defaults to 1080.")

    args = parser.parse_args()

    # Call the main function with command line arguments
    record_stereo(
        output_path=args.output_path, 
        camera_number=None, 
        width=int(args.width), 
        height=int(args.height),
        fps=10,
        save_images=True
        )