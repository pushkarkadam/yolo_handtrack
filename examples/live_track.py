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


sys.path.append('../')

import stereocam as sc
import handtrack as ht

if __name__ == '__main__':

    cam_index = sc.

    live_det = ht.YOLOHandPoseLiveRecord(cam=cam_index, 
                                         fps=10, 
                                         filename="tracks_line8.csv", 
                                         model_path=model_path['n'], 
                                         stereo_frame='left', 
                                         frame_size=(672, 376),
                                         confidence_threshold=0.4, 
                                         **{'verbose': True})
