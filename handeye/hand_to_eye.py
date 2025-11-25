import sys 
import os 
import numpy as np 
import cv2 
import time 
from datetime import datetime 
import pandas as pd 


sys.path.append('../')

import stereocam as sc

def get_image_points(rect_img,
                     chessboard_size=(8,6),
                     square_size=0.039,
                     chessboard_flag=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
                     cornerSubPix_winSize=(11,11),
                     cornerSubPix_zeroZone=(-1,-1),
                     cornerSubPix_criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                     save_path=""
                    ):
    """Returns the image points.

    Parameters
    ----------
    rect_img: numpy.ndarray
        A rectified image.
    chessboard_size: tuple, default, ``(8,6)``
        Size of the inner grid of the chessboard.
        Inner grid points are the corner points of the black squares connecting each other.
    square_size: float, default ``0.039``.
        The size of the square of the chessboard. 
        The default dimension is in meters.
        If other units are chosen, make sure to stay consistent with the units in depth detection as well.
    chessboard_flag: int, default ``cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE``
        Uses the threshold from the enum provided in opencv.
    cornerSubPix_winSize: tuple, default ``(11,11)``
        A tuple of ``int`` that uses the kerner size refined corners.
    cornerSubPix_zeroZone: tupple, default ``(-1, -1)``
        Half of the size of the dead region in the middle of the search zone over which the summation in the formula below is not done. 
        It is used sometimes to avoid possible singularities of the autocorrelation matrix. 
        The value of ``(-1,-1)`` indicates that there is no such a size.
    cornerSubPix_criteria: tuple, default ``(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)``
        A tuple of critereo that is used in fine grained sub pixed calculations.
    save_path=""

    Returns
    -------
    imgpoints: np.array
        A numpy array of the image points.
    rect_img: np.ndarray
        A numpy array of the rectified image with calibration pattern rendered.

    """

    # converting image to gray
    gray_rect = cv2.cvtColor(rect_img, cv2.COLOR_BGR2GRAY)

    # extracting the chessboard size
    ch_r, ch_c = chessboard_size
    
    # Creating object points
    objp = np.zeros((ch_r * ch_c, 3), np.float32)
    
    # reshaping to match t
    objp[:, :2] = np.mgrid[0:ch_r, 0:ch_c].T.reshape(-1, 2)
    
    objp *= square_size
    
    # list to store object points and image points from both cameras
    # 3D real world space of the chess board
    # ``objpoints`` is the list of the list of ``objp`` for each set of stereo images  
    objpoints = []
    imgpoints = []
    
    
    ret, corners = cv2.findChessboardCorners(rect_img, chessboard_size, chessboard_flag)
    
    if ret == True:
        objpoints.append(objp)
        # import pdb;pdb.set_trace()
        corners2 = cv2.cornerSubPix(image=gray_rect, corners=corners, winSize=cornerSubPix_winSize, zeroZone=cornerSubPix_zeroZone, criteria=cornerSubPix_criteria)
        imgpoints.append(corners2)

        cv2.drawChessboardCorners(rect_img, chessboard_size, imgpoints[0], True)
    
    if save_path:
        timestamp = int(time.time())
        cv2.imwrite(os.path.join(save_path, f"pattern_detection_{timestamp}.png"), rect_img)

    return imgpoints[0], rect_img