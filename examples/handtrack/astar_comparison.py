import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time 
import random
import csv
import pandas as pd
import copy
import pickle
import resource
import gc
import tracemalloc

sys.path.append('../..')

import stereocam as sc
import handtrack as ht
import seamdet as sd
import astar


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser(description="A* comparison")
    parser.add_argument('-p', '--path')
    parser.add_argument('-w', '--welds', nargs='+', default=['line', 'curve', 'saw_tooth'], help="Pass a list of items separately by spaces such as: --welds line curve saw_tooth")

    args = parser.parse_args()

    # Extracting command line variables
    data_path = str(args.path)
    weldment_types = list(args.welds)

    data_dict = {'exp_ID': [],
             'seam_type': [],
             'nodes_visited': [],
             'num_coords': [],
             'time': [],
             'memory': []
            }


    for weld_type in weldment_types:
        result_path = os.path.join(data_path, weld_type)

        exp_path = [os.path.join(result_path, res_dir) for res_dir in os.listdir(result_path)]

        for exp in exp_path:
            print(exp)

            # Adding seam type
            data_dict['seam_type'].append(weld_type)

            # Adding the experiment ID
            exp_id = os.path.basename(exp)
                
            data_dict['exp_ID'].append(exp_id)

            I = cv2.imread(os.path.join(exp, 'thinned.png'), 0).astype(np.uint8)

            I = np.where(I>=1, 1, 0)

            coords = sd.path.get_path_coords(I)

            end_nodes = sd.path.get_end_nodes(I, coords)

            rect_tracks_df = pd.read_csv(os.path.join(exp, 'rect_tracks.csv'))

            tracking_ends = sd.path.get_tracking_endpoints(rect_tracks_df)

            start, goal = sd.path.true_end_nodes(end_nodes, tracking_ends)

            grid = np.where(I>=1, 0, 1)

            print(f'start value: {grid[start[::-1]]}')
            print(f'goal value: {grid[goal[::-1]]}')
            print(start)
            print(goal)

            # Staring measuring memory
            tracemalloc.start()

            # starting measuring time
            start_time = time.perf_counter()

            a_star = AStar(goal[::-1], start[::-1])
            path = a_star.search()

            end_time = time.perf_counter()

            current, peak = tracemalloc.get_traced_memory()

            tracemalloc.reset_peak()

            peak_memory_mb = peak / (1024 * 1024)

            time_elapsed = end_time - start_time


            if path:
                data_dict['nodes_visited'].append(len(set(a_star.closed_nodes)))
                data_dict['num_coords'].append(len(path))
            else:
                data_dict['nodes'].append(None)
            
            data_dict['time'].append(time_elapsed)
            data_dict['memory'].append(peak_memory_mb)
            
            print(f'Time elapsed: {(time_elapsed):.4f} s')
            print(f'Peak memory usage: {peak_memory_mb:.4f} MB')
            print(f'Number of nodes: {len(path)}')

            # garbage collection
            gc.collect()

    df = pd.DataFrame(data_dict)

    # Uncomment the following to save the results
    # df.to_csv('a_star_tests.csv', index=False)