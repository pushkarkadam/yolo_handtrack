import os 
import pickle
import numpy as np
import cv2
import time
import copy

import stereocam as sc
import handtrack as ht
import seamdet as sd
import learning as lng


def load_learning_data(data_path, lr_idx, base_idx=0, track_file=None):
    """Loads the data and provides background, base, and test images.
    It also loads the track coordinates.

    Parameters
    ----------
    data_path: str
        Path where the data is stored.
    lr_idx: int
        Learning image index to pick from data.
    base_idx: int
        Base image index with weld information for learning.
    track_file: str, default ``None``
        Name of the file where the track coordinates information is
        stored. This should be a ``.pkl`` file.
        The default name for this file is ``fg_left{idx}_tracks.pkl``.

    Returns
    -------
    bg_image: numpy.ndarray
        Background image.
    base_image: numpy.ndarray
        Base image where the weld seam is detected.
    lr_image: numpy.ndarray
        Image where the weld seam is to be learned.
    tracks: dict
        A dictionary of weld coordinates.
        
    """

    data = lng.matching.MatchingDataCollection(data_path)

    # [0] is for the left stereo image in the tuple (left, right)
    bg_image = data.bg[0]
    base_image = data.fgs[base_idx][0]
    lr_image = data.fgs[lr_idx][0]

    if track_file:
        track_coords_path = os.path.join(data_path, 'fg', track_file)
    else: 
        track_coords_path = os.path.join(data_path, 'fg', f'fg_left{base_idx}_tracks.pkl')

    with open(track_coords_path, 'rb') as f:
        tracks = pickle.load(f)['tracks']

    return bg_image, base_image, lr_image, tracks

def seam_path(base_image, tracks):
    """Detects seam path.

    Parameters
    ----------
    base_image: numpy.ndarray
        Base image where the weld is detected.
    tracks: dict
        A dictionary of x and y coordinates of the track weld seam.

    Returns
    -------
    seam_path_dict: dict
        A dictionary of seam path detected with the search algorithm.
    
    """

    x_track, y_track = tracks['x'], tracks['y']

    boxes = sd.seam.point_box(base_image, x_track, y_track, scaling_factor=10)

    roi_image = sd.seam.region_of_interest(base_image, boxes)

    roi_patches = sd.seam.isolate_point_roi(base_image, boxes)

    patch_edges = sd.seam.roi_edges(roi_patches,
                        blur_n=0,
                        blur_kernel=(3,3)
                       )
    
    # Creating images of zero values
    img_zero = np.zeros(base_image.shape[0:2])
    
    patch_roi_img = sd.seam.patch_roi(img_zero, patch_edges)
    
    # Dilation
    
    dilation_kernel_size = (3,3)
    
    dilation_kernel = np.ones(dilation_kernel_size, np.uint8)
    
    img_dil = cv2.dilate(patch_roi_img, kernel=dilation_kernel, iterations=1)
    

    # 2D Filter
    k = 5
    
    kernel = np.ones((k, k), np.float32) / 25
    
    dst_average = cv2.filter2D(img_dil, -1, kernel)
    
    thinned = cv2.ximgproc.thinning(dst_average.astype(np.uint8))
    
    thinned = sd.seam.remove_smaller_clusters(thinned)
    
    # Extracting points
    # ^^^^^^^^^^^^^^^^^
    row_indices, col_indices = np.nonzero(thinned !=0)

    # Seam path search
    # ----------------

    I = copy.copy(thinned)

    I = np.where(I>=1, 1, 0)
    
    coords = sd.path.get_path_coords(I)
    
    end_nodes = sd.path.get_end_nodes(I, coords)
    
    
    tracking_ends = [(x_track[0], y_track[0]), (x_track[-1], y_track[-1])]
    
    start, goal = sd.path.true_end_nodes(end_nodes, tracking_ends)
    
    start_time = time.time()
    
    seam_path = sd.path.search_path(start, goal, I)
    
    print(f'Time elapsed: {(time.time() - start_time):.4f}s')
    
    path_ancestory = sd.path.get_path_ancestry(seam_path, start, goal)
    
    # seam path
    xl, yl = zip(*path_ancestory)
    
    seam_path_dict = {'x': xl, 'y': yl}

    return seam_path_dict

def load_seam_path(tracks):
    """Streamlines the usage of tracks as the seam coordinates.

    Parameters
    ----------
    tracks: dict or str
        A seam dictionary that consists of the x and y coordinates
        representing the seam or the path where this dictionary 
        is stored as a ``.pkl`` file.

    Returns
    -------
    seam_path: numpy.ndarray
        A numpy array with the seam coordinates.
    
    """

    # If the tracks is a location to pickle file
    if type(tracks) == str:
        with open(track_path, 'rb') as f:
            tracks = pickle.load(f)['tracks']

    xl, yl = tracks['x'], tracks['y']

    seam_path = np.array(list(zip(xl, yl)))

    return seam_path

def learn_detect(bg_image, base_image, lr_image, seam_path):
    """Learns the coordinates.
    
    Parameters
    ----------
    bg_image: numpy.ndarray
        Background image.
    base_image: numpy.ndarray
        Base image where the weld seam is detected.
    lr_image: numpy.ndarray
        Image where the weld seam is to be learned.
    seam_path: numpy.ndarray
        A numpy array with the seam coordinates.

    Returns
    -------
    learned_coords: numpy.ndarray
        Learned coordinates from the base weld seams.
    T_PW: numpy.ndarray
        Transformation matrix from base image to the
        learning image.
    image_dict: dict
        A dictionary of all the images formed in this function.
        
    """
    # background subtraction
    # base image
    base_bg_sub, base_bg_mask = lng.matching.background_subtraction(bg_image, base_image)

    # learning image
    lr_bg_sub, lr_bg_mask = lng.matching.background_subtraction(bg_image, lr_image)

    # Eigenvalues and eigenvectors
    # base image
    centroid1, eigenvalues1, eigenvectors1, T1 = lng.matching.find_object_axes(base_bg_mask)

    # Draw axes
    base_axes = lng.matching.draw_object_axes(
        base_bg_mask,
        centroid1,
        eigenvectors1,
        eigenvalues1,
        primary_color=(0,0,255)
    )

    # learning image
    centroid2, eigenvalues2, eigenvectors2, T2 = lng.matching.find_object_axes(lr_bg_mask)

    lr_axes = lng.matching.draw_object_axes(
        lr_bg_mask,
        centroid2,
        eigenvectors2,
        eigenvalues2,
        primary_color=(0,0,255)
    )

    # Transformations
    # ---------------

    T_PW = lng.matching.image_transformation_matrix(T1, T2)
    
    learned_coords = lng.matching.learned_projections(T_PW, seam_path)

    image_dict = {'base_bg_sub': base_bg_sub,
                  'base_bg_mask': base_bg_mask,
                  'lr_bg_sub': lr_bg_sub,
                  'lr_bg_mask': lr_bg_mask,
                  'base_axes': base_axes,
                  'lr_axes': lr_axes
                 }

    return learned_coords, T_PW, image_dict

def rectangle_object_corners(xm, ym, w, h):
    """Returns the corners of rectangle.

    In order to get started, first make sure
    that the object is centered at the axis coordinates.
    This means that set ``xm = 0`` and ``ym = 0``
    to ensure that object's origin is at the centre.
    
    Paramters
    ---------
    xm: int
        Middle point for x.
    ym: int
        Middle point for y.
    w: int
        Width along x.
    h: int
        Height along y.

    Returns
    -------
    points: numpy.array
        A list of corner points.

    Examples
    --------
    >>> rectangle_object_corners(0, 0, 200, 200)
        
    """
    # Creating points
    points = np.array([[xm - int(w/2), ym + int(h/2)],
                       [xm + int(w/2), ym + int(h/2)],
                       [xm + int(w/2), ym - int(h/2)],
                       [xm - int(w/2), ym - int(h/2)]
                      ])
    
    return points

def generate_mask_image(points, image_size=(376, 672)):
    """Generates an image mask.
    
    Parameters
    ----------
    points: numpy.array
        Points of the polygon of type ``numpy.int32``.
    image_size: tuple, default ``(376, 672)``
        Size of the image.

    """

    bg_mask = np.zeros(image_size, dtype=np.uint8)

    points = points.reshape((-1, 1, 2))

    cv2.fillPoly(bg_mask, [points], color=(255, 255 ,255))

    return bg_mask

def generate_mask_points(points, xm, ym):
    """Generates points for creating image mask.

    This is required to bring the points back from the
    origin of the image.

    Parameters
    ----------
    points: numpy.array
        Points.
    xm: int
        x mid point.
    ym: int
        y mid point.
    
    """
    T_wo = np.array([[1, 0, xm],
                     [0, 1, ym], 
                     [0, 0, 1]])

    # Extracting array size
    n = points.shape[0]

    points_mask = T_wo @ np.column_stack((points, np.ones(n).reshape(n,1))).T

    return points_mask.T[:,:-1]

def sim_transformation_matrix(theta):
    """Simulates a transformation matrix.

    Parameters
    ----------
    theta: float
        Angle of rotation in radians.

    Returns
    -------
    T: numpy.ndarray
        A transformation matrix with pure rotation.

    Examples
    --------
    >>> T = sim_transformation_matrix(np.deg2rad(50))
    
    """

    T = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])

    return T

def transform_object_points(points, T, mid_points, translations=(0, 0), sx=1, sy=1):
    """Transforms the points using T transformation matrix
    and mirrors across the y axis.

    Parameters
    ----------
    points: numpy.ndarray
        Points of the polygon.
    T: numpy.ndarray
        Rotation matrix.
    mid_points: tuple
        A tuple of mid points of type ``int``
    translations: tuple, default ``(0, 0)``
        Translation along x and y axes.
    sx: float
        Scaling in x direction.
    sy: float
        Scaling in y direction.

    Returns
    -------
    nd.array
        A matrix of transformed points.

    Examples
    --------
    >>> T = sim_transformation_matrix(np.deg2rad(50))
    >>> transform_object_points(points, T, (200, 300), translations=(100,0), sx=0.2, sy=0.3)
    
    """

    n = points.shape[0]
    
    # Adding a column of ones 
    p = np.column_stack((points, np.ones(n).reshape(n,1)))

    # Constructing the object to world transformation matrix
    xm, ym = mid_points
    
    T_wo = np.array([[1, 0, xm],
                  [0, 1, ym], 
                  [0, 0, 1]])

    # Scaling matrix
    S = np.array([[sx, 0, 0],
                  [0, sy, 0],
                  [0, 0, 1]])

    tx, ty = translations   
    
    # Translation matrix
    Tr = np.array([[1, 0, tx],
                  [0, 1, ty], 
                  [0, 0, 1]])

    
    # Transforming the object with rotation matrix T and the object to origin matrix.
    # T_wo ensures that the points are projected w.r.t to the world orgin.    
    # S is the matrix that will be used to change the scale
    T_wp = T_wo @ Tr @ T @ S

    # projecting the points
    q = T_wp @ p.T

    return q.T[:,:-1]

def generate_simulated_images(theta, object_shape, scaling=(1,1), translations=(0,0), image_shape=(376, 672)):
    """A combined function that generates the images based on the data
    provided.

    Parameters
    ----------
    theta: float
        Angle of rotation
    object_shape: tuple
        Tuple in the form ``(xm, ym, w, h)``
    scaling: tuple, default ``(1, 1)``
        Scaling values corresponding to ``sx`` and ``sy`` in the
        scaling transformation matrix.
    translations: tuple, default ``(0, 0)``
        Translation values in x and y direction 
        for translation transformation matrix.
    image_shape: tuple, default ``(376, 672)``
        Shape of the image generated.

    Returns
    -------
    base_bg_mask: numpy.ndarray
        Base image used in demonstration.
    lr_bg_mask: numpy.ndarray
        Learning image used to project.

    Examples
    --------
    >>> base_bg_mask, lr_bg_mask = generate_simulated_images(theta=30, object_shape=(200, 200, 100, 200), scaling=(0.5, 0.5), translations=(200, 0))
    
    """
    xm, ym, w, h = object_shape

    sx, sy = scaling

    tx, ty = translations

    points = rectangle_object_corners(0, 0, w, h)

    points_mask = generate_mask_points(points, xm, ym)

    base_bg_mask = generate_mask_image(points_mask.astype(np.int32))

    T = sim_transformation_matrix(np.deg2rad(theta))

    transformed_points = transform_object_points(points, T, (xm, ym), translations=(tx, ty), sx=sx, sy=sy)

    lr_bg_mask = generate_mask_image(transformed_points.astype(np.int32))

    return base_bg_mask, lr_bg_mask