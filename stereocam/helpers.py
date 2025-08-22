import numpy as np 
import cv2
import yaml


def load_calibration_data(calib_file_path):
    """Loads calibration data.
    
    Parameters
    ----------
    calib_file_path: str
        Path where the calibration data is stored
    
    """

    return np.load(calib_file_path)

def colorspace_transform(imageL, imageR, colorspace=cv2.COLOR_BGR2HSV):
    """Transforms the image into defined colorspace.
    
    Parameters
    ----------
    imageL: str
        Image path or ``numpy.ndarray`` image matrix.
    imageR: str
        Image path or ``numpy.ndarray`` image matrix.
    colorspace: int, default ``cv2.COLOR_BGR2HSV``

    Returns
    -------
    imageL: numpy.ndarray
        Rectified left image.
    imageR: numpy.ndarray
        Rectified right image.
        
    """
    if type(imageL) == str and type(imageR) == str:
        imageL = cv2.imread(imageL)
        imageR = cv2.imread(imageR)

    images = [imageL, imageR]

    rect_images = [cv2.cvtColor(image, colorspace) for image in images]

    return rect_images

def clahe_filter(imageL, imageR, clipLimit=2.0, tileGridSize=(8,8)):
    """Applying clahe filter.

    Parameters
    ----------
    imageL: str
        Image path or ``numpy.ndarray`` image matrix.
    imageR: str
        Image path or ``numpy.ndarray`` image matrix.
    
    Returns
    -------
    filtered_images: numpy.ndarray
        Filtered image.
    """
    images = [imageL, imageR]

    filtered_images = []

    for image in images:
        c1, c2, c3 = np.split(image, indices_or_sections=3, axis=2)
        clahe = cv2.createCLAHE(clipLimit,  tileGridSize=tileGridSize)

        # applying the filters
        c1, c2, c3 = [clahe.apply(c) for c in [c1, c2, c3]]

        image_filtered = np.stack((c1, c2, c3), axis=2)

        filtered_images.append(image_filtered)

    return filtered_images

def hsv2gray(image):
    """Converts HSV image to grayscale image.
    
    Parameters
    ----------
    image: numpy.ndarray
        Image in the BGR format.
    
    Returns
    -------
    gray: numpy.ndarray
        Grayscale image.
    
    """
    
    # Converting HSV image to BGR
    bgr = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    # Converting BGR to Gray
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    return gray

def load_stereo_params(file_path):
    """Loads variables from a YAML file.
    
    Parameters
    ----------
    file_path: str
        The path where the ``.yaml`` file is stored.

    Returns
    -------
    dict

    Examples
    --------
    >>> stereo_params = load_stereo_params('config.yaml')

    """
    try:
        with open(file_path, 'r') as file:
            stereo_params = yaml.safe_load(file)
    except Exception as e:
        print(e)
        raise

    return stereo_params

def rectify_points(x, y, K, D, R, P):
    """Rectifies the points from the raw image plane to the rectified image plane.
    
    Paramters
    ---------
    x: numpy.ndarray
        An array of x coordinates.
    y: numpy.ndarray
        An array of y coordinates.
    K: numpy.ndarray
        Camera matrix.
    D: numpy.ndarray
        Distortion matrix
    R: numpy.ndarray
        Rotational matrix.
    P: numpy.ndarray
        Translation vector

    Returns
    -------
    x_rect: numpy.ndarray
        Rectified array of x coordinates.
    y_rect: numpy.ndarray
        Rectified array of y coordinates.

    """

    assert x.shape == y.shape
    
    points = np.column_stack([x, y]).reshape(-1, 1, 2).astype(np.float32)
    points_rect = cv2.undistortPoints(points, K, D, R=R, P=P)

    points_rect = points_rect.reshape(-1, 2)

    x_rect = points_rect[:, 0].astype(np.int32)
    y_rect = points_rect[:, 1].astype(np.int32)
    
    return x_rect, y_rect