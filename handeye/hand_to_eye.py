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

def T_world_to_camera(imgpoints, Q, disparity, chessboard_size=(8,6)):
    """Creates transformation matrix from image points.
    
    Parameters
    ----------
    imgpoints: np.ndarray
        A numpy tensor of size ``(m*n, 1, 2)``, where ``(m,n)`` are ``chessboard_size``.
    Q: np.ndarray
        A reprojection matrix of size ``4 x 4``.
    disparity: np.ndarray
        A disparity map generated from stereo matching.
    chessboard_size: tuple, default ``(8,6)``
        The inner grid points of the chessboard.
    
    Returns
    -------
    cTw: np.ndarray
        A transformation matrix from world to camera.

    """

    m, n = chessboard_size

    # World coordinates axes points
    p1 = 0
    p2 = m * (n - 1)
    p3 = m - 1

    points = [imgpoints[i][0] for i in [p1, p2, p3]]

    disp_points = [disparity[p[1].astype(int), p[0].astype(int)] for p in points]

    image_projection_values = [np.append(p, [d, 1]) for p, d in zip(points, disp_points)]

    X_ = np.matmul(Q, np.array(image_projection_values).T)
    X = (X_/X_[-1])[:-1,:]

    x = X[:,1] - X[:,0]
    y = X[:,2] - X[:,0]
    
    x_hat = x / np.linalg.norm(x)
    y_hat = y / np.linalg.norm(y)

    z_hat = np.cross(x_hat, y_hat)

    # Computing again for orthogonality of y with x and z
    y_hat = np.cross(z_hat, x_hat)

    cTw = np.vstack([x_hat, y_hat, z_hat, X[:,0]]).T
    cTw = np.vstack([cTw, np.array([0,0,0,1])])

    return cTw

def T_world_to_robot(p1, p2, p3):
    """Constructs world to robot matrix from the points.

    Paramters
    ---------
    p1: np.array
        A numpy array for the origin of the world coordinate.
    p2: np.array
        A numpy array of the point along the x-axis of the world coordinate.
    p3: np.array
        A numpy array of the point along the y-axis of the world coordinate.

    Returns
    -------
    rTw: np.ndarray
        A transformation matrix from world to robot.
    
    """

    xr = p2 - p1
    yr = p3 - p1
    
    xr_hat = xr / np.linalg.norm(xr)
    yr_hat = yr / np.linalg.norm(yr)
    
    zr_hat = np.cross(xr_hat, yr_hat)

    # Recomputing y_r to ensure orthogonal with x and y axes.
    y_r = np.cross(zr_hat, xr_hat)

    rTw = np.vstack([xr_hat, yr_hat, zr_hat, p1]).T
    rTw = np.vstack([rTw, np.array([0,0,0,1])])
    
    return rTw

def T_camera_to_robot(cTw, rTw):
    """Constructs camera to robot transformation matrix.
    
    cTw: np.ndarray
        World to camera transformation matrix.
    rTw: np.ndarray
        World to robot transformation matrix.

    Returns
    -------
    rTc: np.ndarray
        Camera to robot transformation matrix.

    """

    rTc = np.matmul(rTw, np.linalg.inv(cTw))

    return rTc

def save_transformation(transformations, file_name='Transformations', save_path='.'):
    """Saves the transformation matrix.
    
    Parameters
    ----------
    transformations: dict
        A dictionary of transformation matrices that needs to be saved.
    file_name: 

    """

    timestamp = int(time.time())

    transformations['timestamp'] = timestamp

    if not file_name:
        file_name = f'Transformations_{timestamp}'

    np.savez(os.path.join(save_path, f'{file_name}.npz'), **transformations)

def robot_projection_evaluation(point_num, imgpoints, Q, disparity, rTc):
    """Runs a test on the evaluation point from the chessboard calibration points detected.
    
    Parameters
    ----------
    point_num: int
        The number of point from a range of ``0 - (m*n - 1)`` of the chessboard point.
    imagepoints: np.ndarray
        Image points from the chessboard corner detection points.
    Q: np.ndarray
        A reprojection matrix of size ``4 x 4``.
    disparity: np.ndarray
        A disparity map generated from stereo matching.
    rTc: np.ndarray
        Camera to robot transformation matrix.

    Returns
    -------
    robot_coords: list
        A list of np.ndarray of ``(x, y, z)`` robot coordinates.

    """

    image_point = imgpoints[point_num][0]

    disparity_value = disparity[image_point[1].astype(int), image_point[0].astype(int)]

    projection_points = np.append(image_point, [disparity_value, 1])

    cX_ = np.matmul(Q, projection_points)
    cX = (cX_ / cX_[-1])

    robot_coords = np.matmul(rTc, cX)

    return robot_coords[:-1]

def robot_points_from_camera(imgpoints, Q, disparity, rTc, checkpoints=[0, 7, 47, 40, 9, 14, 38, 33, 18, 21, 29, 26], save_path=''):
    """Provides a list of all the points in evaluation metric.

    Parameters
    ----------
    imagepoints: np.ndarray
        Image points from the chessboard corner detection points.
    Q: np.ndarray
        A reprojection matrix of size ``4 x 4``.
    disparity: np.ndarray
        A disparity map generated from stereo matching.
    rTc: np.ndarray
        Camera to robot transformation matrix.
    checkpoints: list
        A list of checkpoints from the image points.
        Keep the list empty ``[]`` if the robot coordinates of all ``imgpoints`` are needed.
    save_path: str
        A path to save ending with the ``.csv`` extension for filename. Example: ``~/path/to/filename.csv``.

    Returns
    -------
    robot_coords_list: int
        A list of robot coordinates based on the list of checkpoints provided.
    
    """
    
    if not checkpoints:
        checkpoints = list(range(imgpoints.shape[0]))

    robot_coords_list = [robot_projection_evaluation(i ,imgpoints, Q, disparity, rTc) for i in checkpoints]

    if save_path:
        df = pd.DataFrame(robot_coords_list, columns=['x','y','z'], index=list(range(1, len(checkpoints)+1)))
        df.to_csv(save_path, index=True)
    
    return robot_coords_list