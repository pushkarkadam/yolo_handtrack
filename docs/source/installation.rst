============
Installation
============

Cloning the repository
----------------------

To clone this repository use the following command:

.. code-block:: bash 

    git clone https://github.com/pushkarkadam/yolo_handtrack.git


Creating a virtual environment
------------------------------

Create a virtual environment using the following:

.. code-block:: bash

    python3 -m venv venv

Activate the virtual environment:

.. code-block:: bash 

    source venv/bin/activate

Installing packages
-------------------

First install PyTorch. For this project, a spectic version of PyTorch is used.
Using higher version may cause issue with YOLO.
Also, be mindful about the version for Ultralytics.

Installing Torch 
^^^^^^^^^^^^^^^^

This project uses cpu based detection for handpose.
Torch needs a separate installation.
If installed from ``requirements.txt``, then ``cuda`` libraries will also be installed.
``cuda`` is not used in this project.

Use the following line:

.. code-block:: bash 

    pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cpu

Installing from requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once Torch is installed separately, all the other packages can be installed from ``requirements.txt`` file.

Before installing, make sure your environment is active.

Use the following command to install other packages.

.. code-block:: bash 

    pip install -r requirements.txt