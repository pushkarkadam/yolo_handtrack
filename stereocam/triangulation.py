import numpy as np 
import cv2
import glob
import os
from tqdm import tqdm


def rectify_stereo_image(left_img, right_img, data, save_path=None):
    """Rectifies the image for triangulation.

    Parameters
    ----------
    left_img: numpy.ndarray
        A numpy image or file path to the image.
    right_img: numpy.ndarray
        A numpy image or file path to the image.
    data: dict
        A dictionary of camera parameters or a ``.npy`` file.
    save_path: str, default ``None``
        If path is given, then saves the rectified images to that path.
    
    Returns
    -------
    tuple
        A tuple of ``numpy.ndarray`` left and right rectified stereo image.

    """

    if type(left_img) == str:
        left_img = cv2.imread(left_img)
    
    if type(right_img) == str:
        right_img = cv2.imread(right_img)

    if type(data) == str:
        data = np.load(data, allow_pickle=True).item()

    Kl, Dl, Kr, Dr, R, T, img_size = data['Kl'], data['Dl'], data['Kr'], data['Dr'], \
                                 data['R'], data['T'], data['img_size']

    R1, R2, P1, P2, Q, validRoi1, validRoi2 = cv2.stereoRectify(Kl, Dl, Kr, Dr, img_size, R, T)

    xmap1, ymap1 = cv2.initUndistortRectifyMap(Kl, Dl, R1, P1, img_size, cv2.CV_32FC1)
    xmap2, ymap2 = cv2.initUndistortRectifyMap(Kr, Dr, R2, P2, img_size, cv2.CV_32FC1)

    left_img_rectified = cv2.remap(left_img, xmap1, ymap1, cv2.INTER_LINEAR)
    right_img_rectified = cv2.remap(right_img, xmap2, ymap2, cv2.INTER_LINEAR)

    if save_path:
        left_path = os.path.join(save_path, "left_rect.png")
        right_path = os.path.join(save_path, "right_rect.png")
        cv2.imwrite(left_path, left_img_rectified)
        cv2.imwrite(right_path, right_img_rectified)

    return left_img_rectified, right_img_rectified

def disparity_depth_map(left_img, 
                        right_img, 
                        baseline,
                        data, 
                        algorithm="bm",  
                        save_path=None,
                        wls_lambda=8000,
                        wls_sigma=2.5,
                        **kwargs):
    """Returns disparity and depth map of the image.

    Parameters
    ----------
    left_img: numpy.ndarray
        A numpy image or file path to the image.
    right_img: numpy.ndarray
        A numpy image or file path to the image.
    data: dict
        A dictionary of camera parameters or a ``.npy`` file.
    save_path: str, default ``None``
        If path is given, then saves the rectified images to that path.
    wls_lambda: int, default ``8000``
        Lambda value for weighted least squares filter.
    wls_sigma: float, default ``2.5``
        Sigma value for weighted least squares filter.
    
    Returns
    -------
    tuple
        A tuple of ``numpy.ndarray`` left and right rectified stereo image.
    
    """
    if type(left_img) == str:
        left_img = cv2.imread(left_img)
    
    if type(right_img) == str:
        right_img = cv2.imread(right_img)

    if type(data) == str:
        data = np.load(data, allow_pickle=True).item()

    Kl, Dl, Kr, Dr, R, T, img_size = data['Kl'], data['Dl'], data['Kr'], data['Dr'], \
                                 data['R'], data['T'], data['img_size']


    # focal length
    # Using camera intrinsic matrix Kl and Kr to compute fxl and fxr
    fxl, fxr = Kl[0][0], Kr[0][0]

    focal_length = (fxl + fxr)/2

    if algorithm == "bm":
        stereo_alg = cv2.StereoBM_create(**kwargs)
    else:
        stereo_alg = cv2.StereoSGBM_create(**kwargs)

    disp_map = stereo_alg.compute(cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY), 
                                  cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
                                 ).astype(np.float32) / 16.0


    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo_alg)
    right_matcher = cv2.ximgproc.createRightMatcher(stereo_alg)

    disparity_right = right_matcher.compute(right_img, left_img).astype(np.float32) / 16.0

    # Apply WLS filter
    wls_filter.setLambda(wls_lambda)
    wls_filter.setSigmaColor(wls_sigma)

    filtered_disparity = wls_filter.filter(disp_map, left_img, disparity_map_right=disparity_right)

    disparity_map = np.where(disp_map <= 0, 1e-5, disp_map)

    filtered_disparity = np.where(filtered_disparity <= 0, 1e-5, filtered_disparity)

    depth_map = (baseline * focal_length) / filtered_disparity

    if save_path:
        disp_path = os.path.join(save_path, "disp_map.png")
        filtered_disp_path = os.path.join(save_path, "filtered_disp_map.png")
        depth_path = os.path.join(save_path, "depth_map.png")
        cv2.imwrite(disp_path, disp_map)
        cv2.imwrite(filtered_disp_path, filtered_disparity)
        cv2.imwrite(depth_path, depth_map)

    return disp_map, filtered_disparity, depth_map