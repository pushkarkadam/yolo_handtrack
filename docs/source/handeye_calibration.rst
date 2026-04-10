============================
Hand Eye Calibration Process
============================

This document contains information about getting the robot started in ROS2 humble.
The instructions about starting the ROS2 driver for the UR10e assembly and collecting the data for the robot points
will be mentioned here.

Starting the robot
------------------

When the robot boots up, ``Open > Program > remote_control.urp``.

This will load a program where there is remote control with the remote computer IP address enabled.

Configuring environment
-----------------------

The first step would be to ensure that all the ROS2 workspace is enabled.

Type the following to enable the ROS2 workspace.

.. code-block:: bash

    source /opt/ros/humble/setup.bash && cd Documents/code_project/ros2_ws/ && source install/setup.bash

The above code first enables the ROS2 environment, then, we reach to the local ROS2 workspace ``ros2_ws`` and enable the local workspace.


Starting robot driver in ROS2
-----------------------------

Starting all the ROS2 related items for the calibration.
Execute the following in different terminals.

Create an environment variable for the robot IP address as follows:


.. code-block:: bash

    export ROBOT_IP=196.168.x.x # This should be the robot IP set up

.. code-block:: bash

    ros2 launch robot_cell_control start_robot.launch.py use_fake_hardware:='False' use_fake_sensor:='False' robot_ip:=${ROBOT_IP} initial_joint_controller:=scaled_joint_trajectory_controller kinematics_params_file:="${HOME}/my_robot_calibration.yaml"

Starting MoveIt2
----------------

Start Moveit2 with the following commands in two separate terminals.

.. code-block:: bash

    ros2 launch robot_cell_moveit_config move_group.launch.py


.. code-block:: bash 

    ros2 launch robot_cell_moveit_config moveit_rviz.launch.py

Hand Eye Calibration
--------------------

The hand eye calibration needs to be performed in two stages.

First stage requires collecting the images from the stereo camera of the chessboard.

The calibration board can be found on `Calibration Checkerboard Collection`_.

.. _Calibration Checkerboard Collection: https://markhedleyjones.com/projects/calibration-checkerboard-collection 

Use the default A3 - 40mm squares - 8x6 vertices, 9x7 squares.

Print this on A3 paper.

World to camera
^^^^^^^^^^^^^^^

The first step involves running a code to get world to camera transformation matrix.

This is assuming that the stereo calibration is performed. If the stereo calibration is not performed, then refer to ``stereocam`` module to perform the calibration.

Use the following python code to perform the world to camera. There may likely be Jupyter Notebook for performing this task.

.. code-block:: python

    import sys
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    import open3d as o3d
    import time 
    import copy
    from datetime import datetime
    import pandas as pd
    import yaml

    sys.path.append('../..')

    import stereocam as sc
    import handeye

    # Save paths

    timestamp = int(time.time())

    root_path = '/home/robot1/Documents/hand_eye_calibration'

    save_path = os.path.join(root_path, str(timestamp))

    os.makedirs(os.path.join(save_path), exist_ok=True)

    print(f'Save path: {save_path}')

    # Capture stereo image

    cam_num = sc.capture_images.detect_stereo_camera(camera_name="zed")
    print(f'Camera number: {cam_num}')

    imgL, imgR = sc.capture_images.capture_stereo_image(cam_num, image_resolution=(672, 376))

    plt.imshow(imgL)

    cv2.imwrite(os.path.join(save_path, "left_img.png"), imgL)

    # Stereo camera parameters
    cam_param_dir = "2025-08-26-16-47" # Make sure the path and directory is correct

    camera_params_path = os.path.join("../../data/calib/", cam_param_dir, 'stereo_calib.npz')

    cam_data = sc.helpers.load_calibration_data(camera_params_path)

    f = sc.calibration.get_focal_length(cam_data['Q'])
    baseline = sc.calibration.get_baseline(cam_data['Q'])

    # Loading images

    grayL, grayR = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in [imgL, imgR]]

    # Stereo map generation
    stereoMapL, stereoMapR = sc.depth_estimation.stereo_map(cam_data, image_shape=grayL.shape[::-1])

    # Image rectification
    rectL, rectR = sc.depth_estimation.rectify_images(imgL, imgR, stereoMapL, stereoMapR)

    disparity, camera_projection, depth_map, left_cut = sc.depth_estimation.depth_maps(rectL, 
                                                                                    rectR, 
                                                                                    cam_data['Q'],
                                                                                    dispFactor=12, 
                                                                                    blockSize=5,
                                                                                    minDisparity=0,
                                                                                    disp12MaxDiff=-1,
                                                                                    preFilterCap=30,
                                                                                    uniquenessRatio=1,
                                                                                    speckleWindowSize=100, # range 50-200
                                                                                    speckleRange=1, # range 1 or 2
                                                                                    mode = 0,
                                                                                    image_type='rgb',
                                                                                    remove_stereo_blank=False
                                                                                    )

    plt.imshow(disparity, cmap="gray")
    plt.colorbar()

    cv2.imwrite(os.path.join(save_path,"rectL.png"), rectL) 

    pcd = sc.depth_estimation.point_cloud(rectL, 
                                    depth_limits=(0,1), 
                                    camera_projection=camera_projection, 
                                    depth_map=depth_map,
                                    left_cut=None, 
                                    save_path=save_path, 
                                    pcd_name="point_cloud.ply",
                                    cloud_frame_size=0.05)

    imgpoints, rect_img_render = handeye.get_image_points(rectL, save_path='')   

    cv2.imwrite(os.path.join(save_path, "rect_img_render.png"), rect_img_render)   

    np.savez(os.path.join(save_path, "imgpoints.npz"), imgpoints)

    # World to Camera

    cTw = handeye.T_world_to_camera(imgpoints, cam_data['Q'], disparity)

    handeye.save_transformation(cTw, file_name="cTw", save_path=save_path)

Collecting robot points
^^^^^^^^^^^^^^^^^^^^^^^

This needs to be performed using the robot and ROS2.

In a new terminals make sure to configure the ROS2 environment. Ideally, two new terminals would be better.

In the first terminal type the following code to see the transformation.

.. code-block:: bash

    ros2 run tf2_ros tf2_echo ur10e_base ur10e_tip

This will allow you to see the transformation to get an idea where the tool tip is in the robot space.

For the next terminal, make sure you have access to the path where you saved the world to camera transformation matrix.
This information is available from ``print(f'Save path: {save_path}')`` statement in the code in the above section.

The ROS2 package developed for this task is called ``tf2_data_logger``.

The default name to save the coordinates is ``robot_points.csv``. So, add that after the path name as a command line arguments.

Example: ``/home/robot1/Documents/hand_eye_calibration/1775780943/robot_points.csv``

Once all the information is available, type the following code in the terminal.

.. code-block:: bash

    ros2 run tf2_data_logger tf2_data_saver ur10e_tip ur10e_base /home/robot1/Documents/hand_eye_calibration/1775780943/robot_points.csv

After running this line, move the robot to the three location as mentioned in `this paper`_.

.. _this paper:  https://doi.org/10.3390/computers15010053 

After saving these three points, a csv file named ``robot_points.csv`` will be saved in the hand eye calibration directory location provided the path given in the line above is similar to the location where world to camera transformation is saved.

Camera to robot
^^^^^^^^^^^^^^^

The first step here will be to compute world to robot followed by camera to robot.

After the points are saved, the world to robot transformation matrix needs to be generated.
Use the following python code.

.. code-block:: python

    import sys 
    import os 
    
    sys.path.append('../..')

    import handeye

    save_path = '/home/robot1/Documents/hand_eye_calibration/1775780943'

    p1, p2, p3 = handeye.read_robot_calibration_points(save_path)

    rTw = handeye.T_world_to_robot(p1,p2,p3)

    cTw = np.load(os.path.join(save_path, 'cTw.npz'))['cTw']

    rTc = handeye.T_camera_to_robot(cTw, rTw)

    handeye.save_transformation(rTc, file_name="rTc", save_path=save_path)


This completes robot hand-to-eye calibration process.


Error Evaluation
^^^^^^^^^^^^^^^^

The evaluation for the error is measured by following python code.

.. code-block:: python 

    import sys
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    import open3d as o3d
    import time 
    import copy
    from datetime import datetime
    import pandas as pd
    import yaml

    sys.path.append('../..')

    import stereocam as sc
    import handeye


    # Getting stereo paramters
    cam_param_dir = "2025-08-26-16-47"
    camera_params_path = os.path.join("../../data/calib/", cam_param_dir, 'stereo_calib.npz')
    cam_data = sc.helpers.load_calibration_data(camera_params_path)

    # Path where the camera data is stored
    save_path = '/home/robot1/Documents/hand_eye_calibration/1775780943'

    disparity = cv2.imread(os.path.join(save_path, 'disparity.png'), 0)

    rTc = handeye.load_hand_to_eye_matrix(os.path.join(save_path, 'rTc.npz'))

    imgpoints = np.load(os.path.join(save_path, 'imgpoints.npz'))

    robot_coords_list = [handeye.robot_projection_evaluation(i ,imgpoints, cam_data['Q'], disparity, rTc) for i in [0, 40, 7]]

    cam_coords, robot_coords = zip(*robot_coords_list)

    p1, p2, p3 = handeye.read_robot_calibration_points(save_path)

    xrms, yrms, zrms = handeye.reprojection_error((p1,p2,p3), robot_coords, save_path=save_path)

    print(f'x: {xrms}')
    print(f'y: {yrms}')
    print(f'z: {zrms}')

    # Checkpoint evaluation of 12 points

    checkpoints = [0, 7, 47, 40, 9, 14, 38, 33, 18, 21, 29, 26]
    robot_coords = handeye.robot_points_from_camera(imgpoints, cam_data['Q'], disparity, rTc, checkpoints, save_path=save_path)

    # Saves a complete 12 checkpoint evaluation coordinates
    df_image_cam_robot = handeye.image_camera_robot_coords(imgpoints, cam_data['Q'], disparity, rTc, checkpoints, save_path=save_path)


The last part in the above-mentioned code provides 12 checkpoints and it will be saved as a csv file titled ``robot_coords.csv``.
ROS2 will read these points obtained from camera to robot transformation to reach these points in the robot space.

Evaluation
^^^^^^^^^^

This part will deal with ROS2 again. 
Here, we use 12 evalution points and expect the robot to follow a path to reach to them.

This is performed using custom ROS2 package called ``robot_cell_path_planning``.

In a new terminal after the environment is configured, type the following.

.. code-block:: bash 

    ros2 run robot_cell_path_planning calibration_path /home/robot1/Documents/hand_eye_calibration/1775780943/robot_coords.csv


Before clicking on ``Next`` in Rviz Gui as indicated by the lines in the terminal, make sure the robot has remove control switched on.

After executing this code, observe the MoveIt2 Rviz2 window will have changed. Now, you can click on ``Next`` and the robot will perform the path planning and execution.


