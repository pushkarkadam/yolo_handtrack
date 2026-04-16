================================
Seam Detection Robot Experiments
================================


Directory Structure
-------------------

The following directory structure is created under the directory ``${HOME}/robot_seam_exp``

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

Browse to the ``examples/handtrack`` directory from the root of this repository.

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