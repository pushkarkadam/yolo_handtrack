================================
Seam Detection Robot Experiments
================================


Directory Structure
-------------------

The following directory structure is created under the directory ``${HOME}/Documents/robot_seam_exp``

.. code:: bash 

    .
    ├── hand_eye_calibration
    │   └── 1775780943
    ├── seam_line_exp
    │   └── 2026-04-16-18-30
    └── stereo_calibration
        └── 2025-08-26-16-47



Make sure to add the stereo calibration results subdirectory inside the ``stereo_calibration`` directory.

Seam Line Detection
-------------------

Browse to the ``examples/handtrack`` directory from the root of this repository as follows:

.. code-block:: bash

    cd examples/handtrack

To run the hand tracking to seam line detection, run the following:

.. code-block:: bash 

    python seam_detection.py -sp ${HOME}/Documents/robot_seam_exp/seam_line_exp \
    -c 0.8 \
    -cam ${HOME}/Documents/robot_seam_exp/stereo_calibration/2025-08-26-16-47


Seam path search
----------------

Find the seam line path by running the following code:

.. code-block:: bash 

    python seam_path_search.py \
    -p ${HOME}/Documents/robot_seam_exp/seam_line_exp/2026-04-16-18-30 \
    -cam ${HOME}/Documents/robot_seam_exp/stereo_calibration/2025-08-26-16-47 \
    -hp ${HOME}/Documents/robot_seam_exp/hand_eye_calibration/1775780943


Seam line evaluation 
--------------------

To evaluate the seam line path along with the ground truth data run the following:

.. code-block:: bash 

    python checkpoint_evaluation.py \
    -p ${HOME}/Documents/robot_seam_exp/seam_line_exp/2026-04-16-18-30 \
    -b 1

Robot Path implementation
-------------------------

To implement this, we use ROS2 path following package that we developed.

This will be perfomed using ROS2 package called ``robot_cell_path_planning``.

Starting the robot
^^^^^^^^^^^^^^^^^^

When the robot boots up, ``Open > Program > remote_control.urp``.

This will load a program where there is remote control with the remote computer IP address enabled.

Configuring environment
^^^^^^^^^^^^^^^^^^^^^^^

The first step would be to ensure that all the ROS2 workspace is enabled.

Type the following to enable the ROS2 workspace.

.. code-block:: bash

    source /opt/ros/humble/setup.bash && cd Documents/code_project/ros2_ws/ && source install/setup.bash

The above code first enables the ROS2 environment, then, we reach to the local ROS2 workspace ``ros2_ws`` and enable the local workspace.


Starting robot driver in ROS2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Starting all the ROS2 related items for the calibration.
Execute the following in different terminals.

Create an environment variable for the robot IP address as follows:


.. code-block:: bash

    export ROBOT_IP=196.168.x.x # This should be the robot IP set up

.. code-block:: bash

    ros2 launch robot_cell_control start_robot.launch.py use_fake_hardware:='False' use_fake_sensor:='False' robot_ip:=${ROBOT_IP} initial_joint_controller:=scaled_joint_trajectory_controller kinematics_params_file:="${HOME}/my_robot_calibration.yaml"

Starting MoveIt2
^^^^^^^^^^^^^^^^

Start Moveit2 with the following commands in two separate terminals.

.. code-block:: bash

    ros2 launch robot_cell_moveit_config move_group.launch.py


.. code-block:: bash 

    ros2 launch robot_cell_moveit_config moveit_rviz.launch.py

Robot Path following
^^^^^^^^^^^^^^^^^^^^

Here, the robot coordinate saved in `Seam path search`_ will be used as an input
for the robot path.

In a new terminal type the following:

.. code-block:: bash 

    ros2 run robot_cell_path_planning \
    weld_path \
    ${HOME}/Documents/robot_seam_exp/seam_line_exp/2026-04-16-18-30/seam_path_robot_coords.csv

.. warning:: 

    If the depth estimation is not done properly then those points are eliminated when determining the robot path.
    In this case, observe disparity map to make sure that depth estimation is done properly.