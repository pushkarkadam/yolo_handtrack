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


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser(description="Checkpoint Evaluation")
    parser.add_argument('-p', '--path')
    parser.add_argument('-b', '--box', default=1, type=int, help="Bounding box size = 3 * box input. 3 pixel box.")
    
    args = parser.parse_args()

    # Extracting command line variables
    track_path = str(args.path)
    box_extension = int(args.box)

    image_path = os.path.join(track_path, 'rectL.png')

    rectL = cv2.imread(image_path)

    checkpoint_save_path = os.path.join(track_path, 'checkpoints')

    data = sd.path.point_plotter(image_path, checkpoint_save_path, ["checkpoints"], ["x"], ["r"])

    checkpoints = data['checkpoints']

    # Extracting seam path
    seam_path_file = os.path.join(track_path, "seam_path_coords.pkl")

    with open(seam_path_file, 'rb') as f:
        seam_path_dict = pickle.load(f)

    xl, yl = seam_path_dict['x'], seam_path_dict['y']

    xc, yc = checkpoints['x'], checkpoints['y']

    chk_pts = [(x, y) for x, y in zip(xc, yc)]

    visited_checkpoints, passing_ratio = sd.path.evaluate_checkpoints(chk_pts, seam_path_dict, box_extension)

    print(f'Passing ratio: {passing_ratio}')

    not_visited_chk = list(set(chk_pts) - set(visited_checkpoints))

    xv, yv = zip(*visited_checkpoints)
    if not_visited_chk:
        xnv, ynv = zip(*not_visited_chk)
    else:
        xnv, ynv = [], []

    # Saving evaluation results
    evaluation_metric = {'visited_checkpoints': visited_checkpoints,
                         'not_visited_checkpoints': not_visited_chk,
                         'passing_ratio': passing_ratio}

    evaluation_metric_path = os.path.join(track_path, 'evaluation_metric_data.pkl')

    with open(evaluation_metric_path, 'wb') as f:
        pickle.dump(evaluation_metric, f)

    evaluation_metric_result = {'passing_ratio': passing_ratio}

    evaluation_yaml_path = os.path.join(track_path, 'evaluation_data.yml')

    with open(evaluation_yaml_path, 'w') as f:
        yaml.dump(evaluation_metric_result, f)

    plt.imshow(rectL)
    plt.plot(xl, yl, color='purple', alpha=0.5, label='Seam Path')
    plt.scatter(xv, yv, color='g', marker='D', label='Visited checkpoints')
    plt.scatter(xnv, ynv, color='r', marker='X', label='Not visited checkpoints')
    plt.legend()
    plt.axis("off")
    plt.savefig(os.path.join(track_path, 'checkpoint_path.png'), bbox_inches='tight', pad_inches=0)



