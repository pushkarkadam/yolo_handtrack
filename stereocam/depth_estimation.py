import numpy as np 
import cv2
import glob
import os
import open3d as o3d
import sys 

sys.path.append('../')
from stereocam.helpers import hsv2gray


def stereo_map(calib_data, image_shape=(1920, 1080)):
    """Generates stereo maps.
    
    Parameters
    ----------
    calib_data: str
        Calibration data.
        This parameter can be either a ``str`` or ``dict``.
        If the parameter is ``str``, then it will be read using ``load_calibration_data`` function
    image_shape: tuple, default ``(1920, 1080)``
        Shape of the image.
        Although the image shape can be manually added, it is wise to extract the shape from the image
        when implementing in the pipeline.

    Returns
    -------
    stereoMapL: tuple
        A tuple of stereo map for remaping left image.
    stereoMapR: tuple
        A tuple of stereo map for remaping right image.
    
    """

    if type(calib_data) == str:
        calib_data = load_calibration_data(calib_data)

    # Extracting the calibration data from the dictionary
    mtxL, distL = calib_data["mtxL"], calib_data["distL"]
    mtxR, distR = calib_data["mtxR"], calib_data["distR"]
    R, T = calib_data["R"], calib_data["T"]
    R1, R2, P1, P2, Q = calib_data["R1"], calib_data["R2"], calib_data["P1"], calib_data["P2"], calib_data["Q"]

    stereoMapL = cv2.initUndistortRectifyMap(cameraMatrix=mtxL, 
                                             distCoeffs=distL, 
                                             R=R1, 
                                             newCameraMatrix=P1, 
                                             size=image_shape, 
                                             m1type=cv2.CV_16SC2)
    
    stereoMapR = cv2.initUndistortRectifyMap(cameraMatrix=mtxR, 
                                             distCoeffs=distR, 
                                             R=R2, 
                                             newCameraMatrix=P2, 
                                             size=image_shape, 
                                             m1type=cv2.CV_16SC2)

    return stereoMapL, stereoMapR

def rectify_images(imageL, imageR, stereoMapL, stereoMapR, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0):
    """Rectifies the images.

    Parameters
    ----------
    imageL: str
        Image path or ``numpy.ndarray`` image matrix.
    imageR: str
        Image path or ``numpy.ndarray`` image matrix.
    stereoMapL: numpy.ndarray
        Map for rectification of left image.
    stereoMapR: numpy.ndarray
        Map for rectificatrion of right image.
    interpolation: int, default ``cv2.INTER_LINEAR``
        Interpolation method for remapping.
    borderMode: int, default ``cv2.BORDER_CONSTANT``
        Boder mode for the image upon rectification.
    borderValue: int, default ``0``
        Border value.
        When ``0``, the region will have zero values therefore black patches.
        
    
    """
    
    if type(imageL) == str and type(imageR) == str:
        imageL = cv2.imread(imageL)
        imageR = cv2.imread(imageR)

    rectL = cv2.remap(src=imageL, 
                      map1=stereoMapL[0],
                      map2=stereoMapL[1],
                      interpolation=interpolation,
                      borderMode=borderMode,
                      borderValue=borderValue
                     )
    
    rectR = cv2.remap(src=imageR,
                      map1=stereoMapR[0],
                      map2=stereoMapR[1],
                      interpolation=interpolation,
                      borderMode=borderMode,
                      borderValue=borderValue
                     )

    return rectL, rectR

def depth_maps(imageL, 
               imageR, 
               Q,
               image_type='hsv',
               dispFactor=1, 
               blockSize=5, 
               minDisparity=500,
               disp12MaxDiff=-1,
               preFilterCap=30,
               uniquenessRatio=0,
               speckleWindowSize=100, # range 50-200
               speckleRange=2, # range 1 or 2
               mode = 0,
               remove_stereo_blank=True
              ):
    """Disparity map generation.

    Some paramters provided in this function are taken from opencv method for
    stereo disparity estimation using SGBM method.

    Refer to the `OpenCV documentation`_

    .. _OpenCV documentation: https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html
    
    Parameters
    ----------
    imageL: numpy.ndarray
        Left image from the stereo camera. Formats: hsv, bgr, or rgb
    imageR: numpy.ndarray
        Right image from the stereo camera. Formats: hsv, bgr, or rgb
    Q: numpy.ndarray
        Re-projection matrix. Available from camera calibration parameters.
    dispFactor: int, default ``1``
        Disparity factor to be multiplied by ``16`` in the code.
    blockSize: int
        The block size for scanning.
    minDisparity: int
        Minimum number of disparity.
        Can be calculated by knowing the approximate distance of the background object.
    disp12MaxDiff: int, default ``-1``
        Maximum allowed difference (in integer pixel units) in the left-right disparity check. Set it to a non-positive value to disable the check.
    preFilterCap: int, default ``30``
        Truncation value for the prefiltered image pixels.
    uniquenessRatio: int, default ``0``
        Enforces the requirement that the match value for the current pixel is more
        than the minimum match value observed by some margin. Range: ``5 ~ 15``.
    speckleWindowSize: int, default ``100``
        Maximum size of smooth disparity region. Range: ``50 ~ 200``
    speckleRange: int, default ``2``
        Maximum disparity variation. Range: ``1 ~ 2``
    mode: int, default ``0``
        The method used to compute.
    image_type: str, default ``'hsv'``
        Image type as the input to use correct conversion to gray scale.
    remove_stereo_blank: bool, default ``True``
        Removes the area where there is no stereo matching available.
    
    Returns
    -------
    disparity: numpy.ndarray
        Disparity matrix where each element consists of the disparity values.
    camera_projection: numpy.ndarray
        A 3D tensor where each channel consists information about the x, y, and z
        coordinates with respect to the left camera frame.
    depth_map: numpy.ndarray
        A matrix that shows depth value of each pixel in left camera frame.
    left_cut: int
        The number of rows that will be elimated using ``minDisparity + 16 * dispFactor``

    """
    stereo_images = [imageL, imageR]
    
    # converting the images
    if image_type == 'hsv':
        grayL, grayR = [hsv2gray(image) for image in stereo_images]
    elif image_type == 'bgr':
        grayL, grayR = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in stereo_images]
    elif image_type == 'rgb':
        grayL, grayR = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in stereo_images]
    else:
        grayL, grayR = stereo_images

    # creating a stereo object
    stereo = cv2.StereoSGBM.create(
        minDisparity=minDisparity,
        numDisparities=16 * dispFactor,
        blockSize=blockSize,
        P1=8 * 3 * blockSize**2,
        P2=32 * 3 * blockSize**2,
        disp12MaxDiff=disp12MaxDiff,
        preFilterCap=preFilterCap,
        uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize, # range 50-200
        speckleRange=speckleRange, # range 1 or 2
        mode=mode
    )
    
    # Computing disparity
    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

    # Replacing the zero values with smallest values
    disparity[disparity <=0] = 0.1

    # Reprojecting the disparity map to camera coordinates
    camera_projection = cv2.reprojectImageTo3D(disparity, Q)

    # Extracting last channel of the camera projection which is the z axis for depth
    depth_map = camera_projection[:,:,-1]

    # Elimating the blank area on the left side
    left_cut = minDisparity + 16 * dispFactor

    if remove_stereo_blank:
        disparity = disparity[:, left_cut:]
        camera_projection = camera_projection[:, left_cut:, :]
        depth_map = depth_map[:, left_cut:]

    return disparity, camera_projection, depth_map, left_cut

def point_cloud(image,
                depth_limits,
                camera_projection,
                depth_map,
                left_cut,
                image_type='bgr',
                save_path=None,
                pcd_name="point_cloud.ply",
                visualize=True,
                cloud_frame=True,
                cloud_frame_size=0.5
               ):
    """Generates and saves point cloud.
    
    Parameters
    ----------
    image: numpy.ndarray
        Color image used. Ideally left image of the stereo camera.
        The image should be of the camera whose coordinate frames are primary frame.
    depth_limits: tuple
        Limits of the depth to be bounded.
    camera_projection: numpy.ndarray
        A 3D tensor where each channel consists information about the x, y, and z
        coordinates with respect to the left camera frame.
    depth_map: numpy.ndarray
        A matrix that shows depth value of each pixel in left camera frame.
    image_type: str, default ``bgr``
        The channels of the image specified.
    save_path, str, default ``None``
        The path to save the point cloud.
    pcd_name: str, default ``'point_cloud.ply'``
        Name of the point cloud file.
    visualize: bool, default ``True``
        Visualizes the point cloud.
    cloud_frame: bool, default ``True``
        Visualise the camera frame.
    cloud_frame_size: float, default ``0.5``
        Camera frame size in point cloud visualisation.

    Returns
    -------
    open3d.cpu.pybind.geometry.PointCloud

    """

    if image_type == 'bgr':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image[:, left_cut:, :]

    # Normalize to [0, 1]
    colors = image.reshape(-1, 3) / 255.0
    points = camera_projection.reshape(-1, 3)

    colors = colors.astype(np.float32)
    points = points.astype(np.float32)

    # lower and upper limit
    lower_lim, upper_lim = depth_limits

    # Creating a mask as per the depth limit set
    valid_mask = (depth_map > lower_lim) & (depth_map < upper_lim)

    valid_colors = colors[valid_mask.ravel()]
    valid_points = points[valid_mask.ravel()]

    # Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(valid_colors)
    pcd.points = o3d.utility.Vector3dVector(valid_points)

    if save_path:
        o3d.io.write_point_cloud(os.path.join(save_path, pcd_name), pcd)

    vis_data = [pcd]

    if visualize:
        if cloud_frame:
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=cloud_frame_size, origin=[0, 0, 0])
            vis_data.append(axis)
        o3d.visualization.draw_geometries(vis_data)

    return pcd

def pick_points(pcd):
    r"""Provides the coordinate information by clicking on the points.
    The output will be printed in the terminal window when the points are clicked.
    The return list will three points that will provide the index of the point
    if the ``depth_map`` matrix  was flatten such as ``depth_map.ravel()``.

    Use the following procedure when selecting the point:

    - Please pick at least three correspondences using [shift + left click]
    - Press [shift + right click] to undo point picking
    - After picking points, press 'Q' to close the window

    Parameters
    ----------
    pcd: open3d.cpu.pybind.geometry.PointCloud
        Point cloud data
    
    """
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()

def approximate_min_disparity(approx_distance, Q):
    """Estimates the approximate minimum disparity value.

    This method is useful when a distance can be estimated for the object.
    This is performed when setting up the camera system.
    
    Parameters
    ----------
    approx_distance: float
        Approximate distance of the object of interest observed.
    Q: numpy.ndarray
        Reprojection matrix.

    Returns
    -------
    disp_value: float
        Disparity value computed from the reprojection matrix.
        
    """
    
    # Creating a vector of size 4 x 1 
    x_vec = np.array([1,1,approx_distance, 1])

    # Calculating inverse of reprojection matrix Q
    Qinv = np.linalg.inv(Q)

    # Multiplying Q_inv and x_vec to get homogeneous coordinate
    xh = np.dot(Qinv, x_vec)

    # Dividing the homogenous scaling factor
    x = (xh / xh[-1])[:-1]

    # extracting the disparity value from the x vector
    disp_value = x[-1]

    return disp_value

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

def image_points_to_camera(x_rect, y_rect, left_cut, disparity, Q, max_depth=1):
    """Projects the image plane points that are rectified to the points in
    camera frame.
    
    Parameters
    ----------
    x_rect: numpy.ndarray
        An array of rectified x coordinates.
    y_rect: numpy.ndarray
        An array of rectified y coordinates.
    left_cut: int
        Value where the disparity image is cut with the blank area.
    disparity: numpy.ndarray
        Disparity map.
    Q: numpy.ndarray
        Projection matrix.
    max_depth: int, default ``1``
        Maximum depth for visualisation.
    
    Returns
    -------
    points3d: numpy.ndarray
        A point3d array where the image coordinates are projected into camera coordinates.
    """

    image_points = [(xi - left_cut, yi) for xi, yi, in zip(x_rect, y_rect)]
    
    disp_points = [disparity[i] for i in image_points]

    object_points = [np.array([c[0] + left_cut, c[1], d, 1]) for c, d in zip(image_points, disp_points)]

    object_3d = [np.matmul(Q, p) for p in object_points]

    object_3d = [op/op[-1] for op in object_3d]

    # filtering points to exclude those beyond estimated depths
    filtered_points3d = [i for i in object_3d if i[-2] < max_depth]

    points3d = np.array([o[:-1].tolist() for o in filtered_points3d])

    return points3d

def visualise_points(pcd, points3d, color=[1, 0, 0]):
    """Visualises the given set of points in point clouds.
    
    Parameters
    ----------
    pcd: open3d.cpu.pybind.geometry.PointCloud
        Point cloud data
    points3d: ndarray
        An array of size ``
    """ 

    points_to_add = o3d.geometry.PointCloud()
    points_to_add.points = o3d.utility.Vector3dVector(points3d)
    points_to_add.paint_uniform_color(color)
    o3d.visualization.draw_geometries([points_to_add, pcd])