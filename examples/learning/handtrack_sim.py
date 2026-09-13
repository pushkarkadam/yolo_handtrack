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
import matplotlib.pyplot as plt
import yaml

sys.path.append('../..')

import stereocam as sc
import handtrack as ht
import seamdet as sd
import learning as lng 


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser(description="Simulating the hand track for fast experiments.")
    parser.add_argument('-p', '--path', help="Path where the image is stored")
    parser.add_argument('-i', '--img',  help="Image name e.g. left_fg.png")
    # parser.add_argument('-b', '--box', default=1, type=int, help="Bounding box size = 3 * box input. 3 pixel box.")
    
    args = parser.parse_args()

    # Extracting command line variables
    data_path = str(args.path)
    image_name = str(args.img)

    # Simulating handtracking

    image_path = os.path.join(data_path, image_name)

    tracking_save_path = os.path.join(data_path, f'{image_name[:-4]}_tracks')

    data = sd.path.point_plotter(image_path, tracking_save_path, ['tracks'], ['x'], ['r'])

    track_coords = data['tracks']

    # print(track_coords)