import numpy as np 
import cv2
import glob
import os
from tqdm import tqdm


def read_chess_board(file_path, pattern_dim=(7, 3), image_format="png", display_rendered=False):
    """Function to read the features of the chess board for calibration.

    Parameters
    ----------
    file_path: str
        Path to images
    pattern_dim: tuple, default ``(7, 3)``
        Dimension of the chess board.
        Make sure to subtract the border boxes of the chessboard.
    image_format: str, default ``"png"``
        The format to read the images.
    display_rendered: bool, default ``False``
        Displays rendered images.

    Returns
    -------
    objpoints: list
        A list of ``np.array`` of the world coordinates points.
    imgpoints: list
        A list of ``np.array`` of image coordinates corresponding to the world coordinates.
    
    """
    # pattern dimension
    row_p, col_p = pattern_dim

    # terminaltion criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ..., (6,5,0)
    objp = np.zeros((row_p * col_p, 3), np.float32)
    objp[:,:2] = np.mgrid[0:row_p, 0:col_p].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.

    # 3d point in real world space
    objpoints = []

    # 2d points in image plane
    imgpoints = []

    images = list(sorted(glob.glob(os.path.join(file_path ,'*.' + image_format))))
    
    for fname in tqdm(images, colour="green"):
        img = cv2.imread(fname)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corneres
        ret, corners = cv2.findChessboardCorners(gray, (row_p, col_p), None)

        # If found, and object points, image points (after regining them)
        if ret:
            # Appending the objp to 3d points in real world space
            objpoints.append(objp)

            # Refining the corner location
            refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Appending the refined corners to 2d points in image plane
            imgpoints.append(refined_corners)

            # Draw and display the corners
            if display_rendered:
                cv2.drawChessboardCorners(img, (row_p, col_p), refined_corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)

    cv2.destroyAllWindows()

    return objpoints, imgpoints

# Calibration
# ret, mtx, dist, 

def calibration(file_path, 
                pattern_dim=(7, 3), 
                image_format="png", 
                display_rendered=False, 
                undistort_name="calibresult.png", 
                img_crop=None,
                cam_save_postfix=None
               ):
    """Function to read the features of the chess board for calibration.

    Parameters
    ----------
    file_path: str
        Path to images
    pattern_dim: tuple, default ``(7, 3)``
        Dimension of the chess board.
        Make sure to subtract the border boxes of the chessboard.
    image_format: str, default ``"png"``
        The format to read the images.
    display_rendered: bool, default ``False``
        Displays rendered images.
    undistort_name: str, default ``"calibresult.png"``
        The name of the image to store to check the distortion.
    img_crop: list, default ``None``
        A list of tuples that crops the image.
        e.g. ``[(0,100), (10,100)]`` will crop the image as ``I[0:100, 10:100]``
        If the value is ``None``, then it will not crop the images.
    cam_save_postfix: str, default ``None``
        This is used to save the camera matrix and distortion coefficients.
    
    """
    # pattern dimension
    row_p, col_p = pattern_dim

    # terminaltion criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ..., (6,5,0)
    objp = np.zeros((row_p * col_p, 3), np.float32)
    objp[:,:2] = np.mgrid[0:row_p, 0:col_p].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.

    # 3d point in real world space
    objpoints = []

    # 2d points in image plane
    imgpoints = []

    images = list(sorted(glob.glob(os.path.join(file_path ,'*.' + image_format))))
    
    for fname in tqdm(images, colour="green"):
        img = cv2.imread(fname)

        if img_crop:
            c = img_crop
            img = img[c[0][0]: c[0][1], c[1][0]: c[1][1], :]
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corneres
        ret, corners = cv2.findChessboardCorners(gray, (row_p, col_p), None)

        # If found, and object points, image points (after regining them)
        if ret:
            # Appending the objp to 3d points in real world space
            objpoints.append(objp)

            # Refining the corner location
            refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Appending the refined corners to 2d points in image plane
            imgpoints.append(refined_corners)

            # Draw and display the corners
            if display_rendered:
                cv2.drawChessboardCorners(img, (row_p, col_p), refined_corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Undistortion
    img = cv2.imread(images[0])
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image 
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.putText(dst, images[0], (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(undistort_name, dst)

    # Saving camera matrix and distortion coefficients
    if cam_save_postfix:
        cam = cam_save_postfix
        campar_save_path = os.path.join(file_path, f'cam_cal_parameters_{cam}.npz')
        np.savez(campar_save_path,
                 newcameramtx = newcameramtx,
                 mtx = mtx,
                 dist = dist,
                 rvecs = rvecs,
                 tvecs = tvecs,
                 refined_corners = refined_corners
                )

    # reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error

    print(f'total error: {mean_error/len(objpoints)}')

def stereo_calibration(file_path, pattern_size, chess_box_size=30, save_path=None, save_rendered=None):
    r"""Performs stereo calibration.
    
    Saves the camera matrix  as a dictionary and can be loaded using ``numpy.load()`` function.
    
    Paramteres
    ----------
    file_path: str
        Path to where the stereo images are stored.
        The storage format must be ``file_path/stereo_left`` and file_path/stereo_right``.
        The ``file_path`` must comprise of two subdirectories ``stereo_left`` and ``stereo_right``.
    patter_size: tuple
        A tuple of the pattern size.
        Use one size less for pattern.
        For a chess board of ``9 x 8`` pattern, use the input as ``(8, 7)``.
    chess_box_size: int, default ``30``
        The chess box size of each grid. 
        The default value is ``30 cm``
    save_path: str, default ``None``
        If the save path is provided, then the parameters are saved in this location
        otherwise the parameters are saved in the same parent directory of ``file_path``.
    save_rendered: str, default ``None``
        Saves the rendered stereo corner matching if the path is provided.

    Examples
    --------
    >>> from stereocam import *
    >>> stereo_calibration(file_path='../images/case2', pattern_size=(8, 4), chess_box_size=30, save_rendered="../images/case2_calib")

    """
    # Creating the path for left and right stereo images
    left_path = os.path.join(file_path, 'stereo_left/*.png')
    right_path = os.path.join(file_path, 'stereo_right/*.png')

    # reading the path of the left and right stereo images
    left_imgs = list(sorted(glob.glob(left_path)))
    right_imgs = list(sorted(glob.glob(right_path)))
    
    # checking if the number of images for left and right stereo are equal
    assert len(left_imgs) == len(right_imgs)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    left_pts, right_pts = [], []
    img_size = None

    pattern_points_vec = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points_vec[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)

    # Multiplying the coordinate indices with chess box grid dimension
    pattern_points_vec = pattern_points_vec * chess_box_size
    
    # We create a list of arrays equivalent to the number of images.
    # Another way of doing this is to append pattern_points in each iteration.
    # so that the left_pts, right_pts, and pattern_points list have same size.
    # This will save the mismatch error during calibration.
    # however, if you are sure that all the images are clearn and their
    # corners are detected, then, multiplying pattern_points with number
    # of images saves appending operations.
    
    # pattern_points = [pattern_points_vec] * len(left_imgs)
    
    # creating pattern_points as an empty list to populate later.
    pattern_points = []

    for idx, (left_img_path, right_img_path) in enumerate(tqdm(zip(left_imgs, right_imgs))):
        # Reading color images
        I_left = cv2.imread(left_img_path)
        I_right = cv2.imread(right_img_path)

        # Again reading image from source using grayscale
        # The image values will be different if directly read as grayscale vs converted
        # using cv2.COLOR_BGR2GRAY
        left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)
        
        # obtaining image size
        if img_size is None:
            img_size = (left_img.shape[1], left_img.shape[0])
        
        # Finding the chessboard corners for left and right images
        res_left, corners_left = cv2.findChessboardCorners(left_img, pattern_size, None)
        res_right, corners_right = cv2.findChessboardCorners(right_img, pattern_size, None)

        if res_left and res_right:
            pattern_points.append(pattern_points_vec)
        
            # Refining corners
            corners_left = cv2.cornerSubPix(left_img, corners_left, (11, 11), (-1,-1),
                                            criteria)
            corners_right = cv2.cornerSubPix(right_img, corners_right, (11, 11), (-1,-1), 
                                            criteria)
            
            # Appending the corner points
            left_pts.append(corners_left)
            right_pts.append(corners_right)

            # performs rendering and saves the imamge to save_rendered directory
            if save_rendered:
                # Creates directory if it does not exist
                if not os.path.exists(save_rendered):
                    os.makedirs(save_rendered)
                    left_path = os.path.join(save_rendered, 'stereo_left')
                    right_path = os.path.join(save_rendered, 'stereo_right')
                    os.makedirs(left_path)
                    os.makedirs(right_path)

                cv2.drawChessboardCorners(I_left, pattern_size, corners_left, res_left)
                cv2.drawChessboardCorners(I_right, pattern_size, corners_right, res_right)
                cv2.imshow('left_img', I_left)
                cv2.imshow('right_img', I_right)
                cv2.waitKey(500)
                cv2.imwrite(os.path.join(left_path, f"left_img{idx}.png"), I_left)
                cv2.imwrite(os.path.join(right_path, f"right_img{idx}.png"), I_right)

    cv2.destroyAllWindows()

    # Stereo calibration
    err, Kl, Dl, Kr, Dr, R, T, E, F = cv2.stereoCalibrate(
        objectPoints=pattern_points, 
        imagePoints1=left_pts, 
        imagePoints2=right_pts, 
        cameraMatrix1=None, 
        distCoeffs1=None, 
        cameraMatrix2=None, 
        distCoeffs2=None, 
        imageSize=img_size, 
        flags=0
        )


    print(f'Overall projection error: {err}')

    # If the save path is provided otherwise saves at the same location as data.
    if save_path:
        save_path = os.path.join(save_path, 'stereo.npy')
    else:
        save_path = os.path.join(file_path, 'stereo.npy')
    
    # saves the camera parameters.
    np.save(save_path, {
        'Kl': Kl, 
        'Dl': Dl, 
        'Kr': Kr, 
        'Dr': Dr, 
        'R': R, 
        'T': T, 
        'E': E, 
        'F': F, 
        'img_size': img_size, 
        'left_pts': left_pts, 
        'right_pts': right_pts
        })