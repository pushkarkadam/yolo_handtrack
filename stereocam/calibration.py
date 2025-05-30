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

def stereo_calibration(file_path, 
                       chessboard_size, 
                       square_size=0.03,
                       chessboard_flag=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
                       cornerSubPix_winSize=(11,11),
                       cornerSubPix_zeroZone=(-1,-1),
                       cornerSubPix_criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                       calibrateCamera_flags=cv2.CALIB_ZERO_TANGENT_DIST,
                       stereoCalibrate_flags=cv2.CALIB_FIX_INTRINSIC,
                       param_save_path='.',
                       image_limit=10,
                       save_rendered=None
                      ):
    """Calibrates stereo camera.

    For high quality result, use at least 10 images of a ``7 x 8`` or larger chessboard.
    
    Parameters
    ----------
    file_path: str
        Path of the file.
    chessboard_size: str
        Size of the grid of the chessboard.
        For a chess board of ``9 x 8`` pattern, use the input as ``(8, 7)``.
    square_size: float, default ``0.03``.
        The size of the square of the chessboard. 
        The default dimension is in meters.
        If other units are chosen, make sure to stay consistent with the units in depth detection as well.
    chessboard_flag: int, default ``cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE``
        Uses the threshold from the enum provided in opencv.
    cornerSubPix_winSize: tuple, default ``(11,11)``
        A tuple of ``int`` that uses the kerner size refined corners.
    cornerSubPix_zeroZone: tuple, default ``(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)``
        A tuple of critereo that is used in fine grained sub pixed calculations.
    calibrateCamera_flags: int, default ``cv2.CALIB_ZERO_TANGENT_DIST``
        Calibrate camera flags
    stereoCalibrate_flags: int, default ``cv2.CALIB_FIX_INTRINSIC``
        Flags for calibrating stereo camera.
    param_save_path: str, default ``'.'``
        Path to save the calibration parameters.
    image_limit: int, default ``10``
        Limit the images. 
        It turns out that more image increases the reprojection error.
    save_rendered: str, None
        Path to save the rendered images.

    Returns
    -------
    retval: float
        Reprojection error.
        A value less than ``1`` is acceptable. 
        A good result would be to have half a pixel i.e. ``0.5`` error.
    
    """

    # Organising the stereo image paths.
    calib_path = {x: os.path.join(file_path, f"stereo_{x}") for x in ["left", "right"]}

    # Load stereo image pairs
    # Unrapping the paths in tuple
    left_images_path, right_images_path = tuple(calib_path.values())


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

    # 2D image points in the left and right image frame
    # Consists of the list of list of image points per set of stereo images
    imgpoints_left = []
    imgpoints_right = []

    # Creating a list of the images from the path
    left_images = sorted(glob.glob(os.path.join(left_images_path, "*.png")))[:image_limit]
    right_images = sorted(glob.glob(os.path.join(right_images_path, "*.png")))[:image_limit]

    # Detect chessboard corners
    for idx, (left_img, right_img) in enumerate(zip(left_images, right_images)):
        # Reading the images
        imgL = cv2.imread(left_img)
        imgR = cv2.imread(right_img)

        # Convering the images from BGR to Gray
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        # Chessboard corners for left and right images
        retL, cornersL = cv2.findChessboardCorners(image=grayL, 
                                                   patternSize=chessboard_size, 
                                                   flags=chessboard_flag)
        
        retR, cornersR = cv2.findChessboardCorners(image=grayR, 
                                                   patternSize=chessboard_size, 
                                                   flags=chessboard_flag)

        if retL and retR:
            # adding the object points of the current set of images to the list of objpoints for all the images
            objpoints.append(objp)

            # Calculating subpixel to get more accurate result
            cornersL = cv2.cornerSubPix(image=grayL, corners=cornersL, winSize=cornerSubPix_winSize, zeroZone=cornerSubPix_zeroZone, criteria=cornerSubPix_criteria)

            cornersR = cv2.cornerSubPix(image=grayR, corners=cornersR, winSize=cornerSubPix_winSize, zeroZone=cornerSubPix_zeroZone, criteria=cornerSubPix_criteria)

            # Appending list of refined cornerpoints to the imagepoints
            imgpoints_left.append(cornersL)
            imgpoints_right.append(cornersR)
            
            # performs rendering and saves the imamge to save_rendered directory
            if save_rendered:
                left_path = os.path.join(save_rendered, 'stereo_left')
                right_path = os.path.join(save_rendered, 'stereo_right')
                # Creates directory if it does not exist
                if not os.path.exists(save_rendered):
                    # os.makedirs(save_rendered)
                    
                    os.makedirs(left_path)
                    os.makedirs(right_path)
                    
                
                cv2.drawChessboardCorners(image=imgL, patternSize=chessboard_size, corners=cornersL, patternWasFound=retL)
                cv2.drawChessboardCorners(image=imgR, patternSize=chessboard_size, corners=cornersR, patternWasFound=retR)

                cv2.imwrite(os.path.join(left_path, f"left_img{idx}.png"), imgL)
                cv2.imwrite(os.path.join(right_path, f"right_img{idx}.png"), imgR)

    # extracting shape of the image from last grayL in the list
    img_size=grayL.shape[::-1]

    # Calibrating individual camera matrix and distortion coefficients to use later in stereoCalibrate for more precision
    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objectPoints=objpoints, imagePoints=imgpoints_left, imageSize=img_size, cameraMatrix=None, distCoeffs=None, flags=calibrateCamera_flags)
    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objectPoints=objpoints, imagePoints=imgpoints_right, imageSize=img_size, cameraMatrix=None, distCoeffs=None, flags=calibrateCamera_flags)

    # Calibrate stereo
    retval, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(objectPoints=objpoints,
                                                                       imagePoints1=imgpoints_left,
                                                                       imagePoints2=imgpoints_right,
                                                                       cameraMatrix1=mtxL,
                                                                       distCoeffs1=distL,
                                                                       cameraMatrix2=mtxR,
                                                                       distCoeffs2=distR,
                                                                       imageSize=img_size,
                                                                       criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
                                                                       flags=stereoCalibrate_flags)

    print(f"Calibration RMS error: {retval}")

    # Stereo rectification
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix1=mtxL, distCoeffs1=distL, cameraMatrix2=mtxR, distCoeffs2=distR, imageSize=img_size, R=R, T=T)
    # save calibration results
    save_path = os.path.join(param_save_path, 'stereo_calib.npz')
    
    np.savez(save_path, mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR, R=R, T=T, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)
        
    return retval