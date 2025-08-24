import os
import re
import cv2
from torch.utils.data import DataLoader, Dataset
import numpy as np 
import torch
import sys
import pandas as pd

sys.path.append("../")

import handtrack as ht
import stereocam as sc


def list_files(directory):
    """Reads the label files.

    Parameters
    ----------
    directory: str
        Path to the directory

    Returns
    -------
    list
    
    """

    file_names = []

    try:
        files = os.listdir(directory)

        file_names = [f for f in files if os.path.isfile(os.path.join(directory, f))]
    except Exception as e:
        print(e)
        raise
        
    return file_names

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [
        int(text)
        if text.isdigit() else text.lower()
        for text in _nsre.split(s)]

def frames_torch_to_numpy(images):
    """Transforms the images that are stored in ``torch.Tensor`` to a list of numpy array.

    Takes the input of a torch tensor that is stacked together with images of certain batch size
    and breaks them down to a list of numpy array.

    Parameters
    ----------
    images: torch.Tensor
        Torch tensor of size ``(batch, channel, height, width)``
    
    """

    return [images[i].permute(1,2,0).numpy() for i in range(images.shape[0])]

def get_yolo_handpose_model(path):
    """Returns the ``YOLOv8{i}_handpose`` trained model.
    
    Parameters
    ----------
    path: str
        Path to the parent directory of the models.
    """

    model_types = ['n', 's', 'm', 'l', 'x']

    weight_paths = [os.path.join(path, f"yolov8{i}_handpose.pt") for i in model_types]

    model_path  = dict()

    for model_type, weight_path in zip(model_types, weight_paths):
        model_path[model_type] = weight_path

    return model_path

def load_tracks_data(data_path, color_conversion=cv2.COLOR_BGR2RGB):
    """Loads tracking data.
    
    Parameters
    ---------- 
    data_path: str
        Path where the data is stored.
    color_conversion: int, default ``cv2.COLOR_BGR2RGB``
        Conversion code to convert the type of image.
    
    Returns
    -------
    tracks_df: pandas.DataFrame
        A pandas dataframe with tracking information.
    imgL: numpy.ndarray
        Left image of the stereo.
    imgR: numpy.ndarray
        Right image of the stereo.

    """

    tracks_df = pd.read_csv(os.path.join(data_path, 'tracks.csv'))

    imgL, imgR = [cv2.imread(os.path.join(data_path, 'images', f'{img_type}.png')) for img_type in ["left", "right"]]

    return tracks_df, imgL, imgR