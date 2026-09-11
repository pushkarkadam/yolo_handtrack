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

def background_subtraction(bg, fg, kernel_size=(5,5), save_path=''):
    """Performs background subtraction.
    
    Parameters
    ----------
    bg: numpy.ndarray
        A numpy array of colour image representing background without object.
    fg: numpy.ndarray
        A numpy array of colour image with object.
    kernel_size: tuple, default ``(5, 5)``
        Kernel size for erosion morphological operation.
    save_path: str, default ``''``
        Path to save the data.
        
    """

    # converting images to grayscale
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    fg = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference
    difference = cv2.absdiff(bg, fg)

    # Otsu's threshold
    optimal_thresh, mask = cv2.threshold(
        difference, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

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

        return output, object_mask

def order_corners(points):
    """
    Order four points as:
        top-left
        top-right
        bottom-right
        bottom-left
    """

    points = np.asarray(points, dtype=np.float32)

    # Sum and difference of coordinates
    s = points.sum(axis=1)
    d = np.diff(points, axis=1).flatten()

    top_left = points[np.argmin(s)]
    bottom_right = points[np.argmax(s)]

    top_right = points[np.argmin(d)]
    bottom_left = points[np.argmax(d)]

    return np.array([
        top_left,
        top_right,
        bottom_right,
        bottom_left
    ], dtype=np.float32)


def find_rectangle_corners(mask):
    """
    Find the four corners of the largest rectangular object
    in a binary mask.

    Parameters
    ----------
    mask : numpy.ndarray
        Binary image. Object should be white (255),
        background should be black (0).

    Returns
    -------
    corners : numpy.ndarray
        Four corners in order:
        [top-left, top-right, bottom-right, bottom-left]
    """

    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        raise ValueError("No object found in the mask.")

    # Select largest contour
    contour = max(contours, key=cv2.contourArea)

    # Perimeter of contour
    perimeter = cv2.arcLength(contour, True)

    # Approximate contour with polygon
    # Try different epsilon values if necessary
    epsilon = 0.02 * perimeter

    approx = cv2.approxPolyDP(
        contour,
        epsilon,
        True
    )

    # We expect four corners
    if len(approx) != 4:

        # Try a slightly larger approximation
        epsilon = 0.04 * perimeter

        approx = cv2.approxPolyDP(
            contour,
            epsilon,
            True
        )

    if len(approx) != 4:
        raise ValueError(
            f"Could not find exactly four corners. "
            f"Detected {len(approx)} points."
        )

    # Extract coordinates
    corners = approx.reshape(4, 2)

    # Order corners
    corners = order_corners(corners)

    return corners

def find_object_axes(binary_image):
    """
    Find the centroid, eigenvalues, eigenvectors and
    homogeneous transformation matrix of a binary object.

    Paramters
    ---------
    binary_image: np.array
        Binary image as an object mask.
        

    Returns
    -------
    centroid: tuple
        Coordinates of the centroid ``(cx, cy)`` of the object where the coordinate frame will be placed.
    eigenvalues: np.array 
        sorted eigenvalues
    eigenvectors: np.array 
        corresponding eigenvectors
    T: np.array 
        3x3 homogeneous transformation matrix.
        
    """

    # ---------------------------------------------------------
    # 1. Check input
    # ---------------------------------------------------------
    if len(binary_image.shape) != 2:
        raise ValueError("Input image must be a grayscale binary image.")

    # Convert to binary
    mask = binary_image > 0

    # Get coordinates of object pixels
    # OpenCV/image coordinates:
    # x = column
    # y = row
    y, x = np.nonzero(mask)

    if len(x) == 0:
        raise ValueError("No object found in the binary image.")

    # ---------------------------------------------------------
    # 2. Calculate centroid
    # ---------------------------------------------------------
    cx = np.mean(x)
    cy = np.mean(y)

    centroid = np.array([cx, cy])

    # ---------------------------------------------------------
    # 3. Center all object points around centroid
    # ---------------------------------------------------------
    points = np.column_stack((x, y)).astype(np.float64)

    centered_points = points - centroid

    # ---------------------------------------------------------
    # 4. Calculate covariance matrix
    # ---------------------------------------------------------
    covariance = np.cov(centered_points, rowvar=False)

    print("Centroid:")
    print(centroid)

    print("\nCovariance matrix:")
    print(covariance)

    # ---------------------------------------------------------
    # 5. Calculate eigenvalues and eigenvectors
    # ---------------------------------------------------------
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # np.linalg.eigh returns eigenvalues in ascending order.
    # We want the largest eigenvalue first because it corresponds
    # to the object's major axis.
    order = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # ---------------------------------------------------------
    # 6. Make eigenvector directions deterministic
    # ---------------------------------------------------------
    # Eigenvectors can arbitrarily point in either direction.
    # We force the first principal axis to point approximately
    # in the +x direction when possible.

    # NOTE: Not making eigenvalue directions deterministic as
    # forcing eigen vector's direction affects object's true
    # orientation
    
    
    # if eigenvectors[0, 0] < 0:
    #     eigenvectors[:, 0] *= -1

    # Make the second axis form a right-handed 2D coordinate
    # system with the first axis.
    
    # if np.linalg.det(eigenvectors) < 0:
    #     eigenvectors[:, 1] *= -1

    print("\nEigenvalues:")
    print(eigenvalues)

    print("\nEigenvectors:")
    print(eigenvectors)

    # ---------------------------------------------------------
    # 7. Construct the 3x3 transformation matrix
    # ---------------------------------------------------------
    #
    # eigenvectors[:,0] = X axis of object
    # eigenvectors[:,1] = Y axis of object
    #
    # The columns of R are the object axes expressed in
    # image coordinates.
    #
    #                | vx1  vx2  cx |
    # T =            | vy1  vy2  cy |
    #                |  0    0    1 |
    #
    R = np.eye(3)

    R[0:2, 0] = eigenvectors[:, 0]
    R[0:2, 1] = eigenvectors[:, 1]

    T = R.copy()

    T[0, 2] = cx
    T[1, 2] = cy

    print("\n3x3 Transformation matrix:")
    print(T)

    return centroid, eigenvalues, eigenvectors, T


def draw_object_axes(image, centroid, eigenvectors, eigenvalues):
    """
    Draw centroid and PCA axes on the image.

    Parameters
    ----------
    image: np.array
        Image to draw axes.
    centroid: np.array
        Centroid coordinates where the frame rests.
    eigenvectors: np.array
        Eigenvectors show the direction of the axes.
    eigenvalues: np.array
        Eigenvalues show the length of the axes.

    Returns
    -------
    output: np.array
        A numpy array of image with renderings of axes.
        
    """
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    cx, cy = centroid

    # Length of axes for visualization
    scale = 3.0

    # Length based on object size
    length1 = scale * np.sqrt(eigenvalues[0])
    length2 = scale * np.sqrt(eigenvalues[1])

    # Major axis
    p1 = np.array([
        cx - eigenvectors[0, 0] * length1,
        cy - eigenvectors[1, 0] * length1
    ])

    p2 = np.array([
        cx + eigenvectors[0, 0] * length1,
        cy + eigenvectors[1, 0] * length1
    ])

    # Minor axis
    p3 = np.array([
        cx - eigenvectors[0, 1] * length2,
        cy - eigenvectors[1, 1] * length2
    ])

    p4 = np.array([
        cx + eigenvectors[0, 1] * length2,
        cy + eigenvectors[1, 1] * length2
    ])

    # Convert to integer pixel coordinates
    p1 = tuple(np.round(p1).astype(int))
    p2 = tuple(np.round(p2).astype(int))
    p3 = tuple(np.round(p3).astype(int))
    p4 = tuple(np.round(p4).astype(int))

    # Draw axes
    cv2.line(output, p1, p2, (0, 0, 255), 2)  # Major axis
    cv2.line(output, p3, p4, (0, 255, 0), 2)  # Minor axis

    # Draw centroid
    cv2.circle(
        output,
        (int(round(cx)), int(round(cy))),
        5,
        (255, 0, 0),
        -1
    )

    return output