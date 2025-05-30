import os
import re
import cv2
from torch.utils.data import DataLoader, Dataset
import numpy as np 
import torch
import sys

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

def rectify_stereo_images(imageL, imageR, cam_data_path):
    """Rectifies the images.

    Unlike ``rectify_images`` function from ``stereocam`` package,
    this function performs the process of extraction of the stereomap
    and rectifying the images.
    Since these set of steps are provided in the ``stereocam`` package,
    it makes sense to have the function in this module to avoid repeating the
    steps if it is needed in the next function.

    Parameters
    ----------
    imageL: numpy.ndarray
        Left RGB image
    imageR: numpy.ndarray
        Right RGB image
    calib_data: str
        Path to calibration data.

    Returns
    -------
    rectL: numpy.ndarray
        Rectified left image
    rectR: numpy.ndarray
        Rectified right image
    
    """

    calib_data = np.load(cam_data_path, allow_pickle=True)
    
    grayL, grayR = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in [imageL, imageR]]
    
    # Stereo map generation
    stereoMapL, stereoMapR = sc.depth_estimation.stereo_map(calib_data, image_shape=grayL.shape[::-1])

    # Image rectification - hsv
    rectL, rectR = sc.depth_estimation.rectify_images(imageL, imageR, stereoMapL, stereoMapR)

    return rectL, rectR