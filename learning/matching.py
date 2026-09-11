import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import copy

sys.path.append('../')

import stereocam as sc


class MatchingDataCollection:
    """Capture, store, save, and load stereo image matching data.

    This class provides functionality for capturing a single stereo
    background image and multiple stereo foreground image pairs. The
    captured images can be saved to disk and later loaded by providing
    the dataset path when creating a new instance.

    The purpose of this class is to streamline the process of capturing the 
    data for quickly collecting the data from the camera instead of using
    native camera software to manually capture and save it in a desired location.

    The expected directory structure is::

        save_path/
        ├── bg/
        │   ├── bg_left.png
        │   └── bg_right.png
        └── fg/
            ├── fg_left0.png
            ├── fg_right0.png
            ├── fg_left1.png
            ├── fg_right1.png
            └── ...

    Parameters
    ----------
    save_path : str or None, optional
        Path to an existing dataset. If provided, the background and
        foreground images are loaded from this directory. If ``None``,
        a new empty collection is created and the stereo camera is
        initialised.

    Raises
    ------
    FileNotFoundError
        If ``save_path`` is provided but the expected directories or
        image files cannot be found.
    ValueError
        If the dataset contains invalid or incomplete foreground image
        pairs.
    RuntimeError
        If the stereo camera cannot be detected when creating a new
        collection.

    Examples
    --------
    Create a new collection and capture images:

    >>> data = MatchingDataCollection()
    >>> data.captureBG()
    >>> data.captureFG() #If running in jupyter notebook, keep running the same cell
    >>> data.captureFG()
    >>> data.save_images("matching_dataset")

    Load an existing dataset:

    >>> data = MatchingDataCollection("matching_dataset")
    >>> bg_left, bg_right = data.bg
    >>> len(data.fgs)
    2

    The loaded foreground images can be accessed as follows:

    >>> fg_left, fg_right = data.fgs[0]
    """

    def __init__(self, save_path=None):
        """Initialise a new collection or load an existing dataset.

        Parameters
        ----------
        save_path : str or None, optional
            Path to an existing dataset. If provided, the images are
            loaded from disk. If ``None``, a new collection is created
            and the stereo camera is initialised.
        """
        self.fgs = []
        self.bg = None
        self.cam = None

        if save_path is not None:
            self._load_images(save_path)
        else:
            self._initialise_camera()

    def _initialise_camera(self):
        """Initialise the stereo camera.

        Raises
        ------
        RuntimeError
            If the stereo camera cannot be detected.
        """
        try:
            self.cam = sc.capture_images.detect_stereo_camera("zed")
        except Exception as e:
            raise RuntimeError(
                "Failed to detect the stereo camera."
            ) from e

    def captureBG(self):
        """Capture and store a stereo background image.

        The captured left and right images are stored as a tuple in
        ``self.bg``. Calling this method again replaces the previously
        captured background image.

        Examples
        --------
        >>> data = MatchingDataCollection()
        >>> data.captureBG()

        Access the captured images:

        >>> bg_left, bg_right = data.bg
        """
        if self.cam is None:
            raise RuntimeError(
                "Camera is not initialised. "
                "Create the object without a save_path to capture images."
            )

        left, right = sc.capture_images.capture_stereo_image(self.cam)
        self.bg = (left, right)

    def captureFG(self):
        """Capture and store a stereo foreground image pair.

        Each captured foreground image pair is appended to ``self.fgs``.
        Multiple foreground image pairs can be captured by calling this
        method repeatedly.

        Examples
        --------
        >>> data = MatchingDataCollection()
        >>> data.captureFG()
        >>> data.captureFG()

        Check the number of captured foreground pairs:

        >>> len(data.fgs)
        2
        """
        if self.cam is None:
            raise RuntimeError(
                "Camera is not initialised. "
                "Create the object without a save_path to capture images."
            )

        left, right = sc.capture_images.capture_stereo_image(self.cam)
        self.fgs.append((left, right))

    def save_images(self, save_path):
        """Save the captured images to disk.

        The background stereo pair is saved in a ``bg`` subdirectory,
        while all foreground stereo pairs are saved in an ``fg``
        subdirectory.

        Parameters
        ----------
        save_path : str
            Path to the directory where the image dataset will be saved.

        Raises
        ------
        ValueError
            If a background image has not been captured.
        OSError
            If the required directories cannot be created.

        Examples
        --------
        >>> data = MatchingDataCollection()
        >>> data.captureBG()
        >>> data.captureFG()
        >>> data.captureFG()
        >>> data.save_images("matching_dataset")

        This produces::

            matching_dataset/
            ├── bg/
            │   ├── bg_left.png
            │   └── bg_right.png
            └── fg/
                ├── fg_left0.png
                ├── fg_right0.png
                ├── fg_left1.png
                └── fg_right1.png
        """
        if self.bg is None:
            raise ValueError("No background image has been captured.")

        bg_left, bg_right = self.bg

        bg_path = os.path.join(save_path, "bg")
        fg_path = os.path.join(save_path, "fg")

        os.makedirs(bg_path, exist_ok=True)
        os.makedirs(fg_path, exist_ok=True)

        cv2.imwrite(
            os.path.join(bg_path, "bg_left.png"),
            bg_left
        )
        cv2.imwrite(
            os.path.join(bg_path, "bg_right.png"),
            bg_right
        )

        for i, (fg_left, fg_right) in enumerate(self.fgs):
            cv2.imwrite(
                os.path.join(fg_path, f"fg_left{i}.png"),
                fg_left
            )
            cv2.imwrite(
                os.path.join(fg_path, f"fg_right{i}.png"),
                fg_right
            )

    def _load_images(self, save_path):
        """Load a previously saved image dataset.

        Parameters
        ----------
        save_path : str
            Path to a directory containing ``bg`` and ``fg``
            subdirectories.

        Raises
        ------
        FileNotFoundError
            If the dataset directory or required background images
            cannot be found.
        ValueError
            If a foreground image does not have a corresponding stereo
            image.
        """
        if not os.path.isdir(save_path):
            raise FileNotFoundError(
                f"Dataset directory does not exist: {save_path}"
            )

        bg_path = os.path.join(save_path, "bg")
        fg_path = os.path.join(save_path, "fg")

        if not os.path.isdir(bg_path):
            raise FileNotFoundError(
                f"Background directory does not exist: {bg_path}"
            )

        if not os.path.isdir(fg_path):
            raise FileNotFoundError(
                f"Foreground directory does not exist: {fg_path}"
            )

        # Load background images
        bg_left_path = os.path.join(bg_path, "bg_left.png")
        bg_right_path = os.path.join(bg_path, "bg_right.png")

        bg_left = cv2.imread(bg_left_path)
        bg_right = cv2.imread(bg_right_path)

        if bg_left is None:
            raise FileNotFoundError(
                f"Could not load background left image: {bg_left_path}"
            )

        if bg_right is None:
            raise FileNotFoundError(
                f"Could not load background right image: {bg_right_path}"
            )

        self.bg = (bg_left, bg_right)

        # Load foreground images
        i = 0

        while True:
            fg_left_path = os.path.join(
                fg_path,
                f"fg_left{i}.png"
            )
            fg_right_path = os.path.join(
                fg_path,
                f"fg_right{i}.png"
            )

            left_exists = os.path.exists(fg_left_path)
            right_exists = os.path.exists(fg_right_path)

            # Stop once there are no more foreground images.
            if not left_exists and not right_exists:
                break

            # Both images must exist for a stereo pair.
            if not left_exists or not right_exists:
                raise ValueError(
                    f"Incomplete foreground image pair at index {i}."
                )

            fg_left = cv2.imread(fg_left_path)
            fg_right = cv2.imread(fg_right_path)

            if fg_left is None:
                raise ValueError(
                    f"Could not load foreground left image: "
                    f"{fg_left_path}"
                )

            if fg_right is None:
                raise ValueError(
                    f"Could not load foreground right image: "
                    f"{fg_right_path}"
                )

            self.fgs.append((fg_left, fg_right))

            i += 1

def background_subtraction(bg, fg, threshold=30, kernel_size=(5,5), threshold_type=cv2.THRESH_BINARY, save_path=''):
    """Performs background subtraction.
    
    Parameters
    ----------
    bg: numpy.ndarray
        A numpy array of colour image representing background without object.
    fg: numpy.ndarray
        A numpy array of colour image with object.
    kernel_size: tuple, default ``(5, 5)``
        Kernel size for erosion morphological operation.
    threshold_type: int, default ``cv2.THRESH_BINARY``.
        This takes the threshold type from the enum available from opencv.
    save_path: str, default ``''``
        Path to save the data.
        
    """

    # converting images to grayscale
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    fg = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference
    difference = cv2.absdiff(bg, fg)

    # threshold operation
    _, mask = cv2.threshold(difference, threshold, 255, threshold_type)

    # Remove small noise
    kernel = np.ones(kernel_size, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find the largest foreground object
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    print(f'Number of contours: {len(contours)}')
    
    if contours:
        # Assume the largest detected object is the foreground object
        largest_contour = max(contours, key=cv2.contourArea)
    
        # Create a clean black mask
        object_mask = np.zeros_like(mask)
    
        # Fill the object
        cv2.drawContours(
            object_mask,
            [largest_contour],
            -1,
            255,
            thickness=cv2.FILLED
        )
    
        # Create completely black output
        output = np.zeros_like(fg)
    
        # Keep only the foreground object
        output[object_mask == 255] = fg[object_mask == 255]

        if save_path:
            image_path = os.path.join(save_path, 'bg_subtracted.png')
            cv2.imwrite(image_path, output)
            print(f'Image saved {image_path}')

        return output