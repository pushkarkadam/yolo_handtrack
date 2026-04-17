import cv2 
import numpy as np 
import matplotlib.pyplot as plt 


def region_box_coords(image, x_vals, y_vals, scaling_factor=100):
    """Returns the region box co-ordinates.
    
    Finds the maximum and minimum x and y co-ordinates
    where the line was traced in the image.
    
    Parameters
    ----------
    image: numpy.ndarray
        An RGB image.
    x_vals: list
        A list of all the x co-ordinates where the finger
        was tracked.
    y_vals: list
        A list of all the y co-ordinates where the finger
        was tracked.
    scaling_factor: int, default ``100``
        The scaling factor is the value in pixel.
        It denotes how further away the padding is 
        required for the image.
        
    Returns
    -------
    tuple
        A tuple of int values in the
        order (x_min, y_min, x_max, y_max)
    
    Examples
    --------
    >>> import numpy as np
    >>> I = np.identity(5)
    >>> x = list(range(1,3))
    >>> y = list(range(1,3))
    >>> boxes = point_box(I, x, y, scaling_factor=100)
    
    """
    # Getting image dimensions
    img_height = image.shape[0]
    img_width = image.shape[1]
    
    x_pad = []
    y_pad = []
    
    # copying for smaller named variable
    sf = scaling_factor
    
    for x, y in zip(x_vals, y_vals):
        # x padding
        x1 = x-sf if x-sf > 0 else 0
        x2 = x+sf if x+sf < img_width else img_width - 1
        
        # y padding
        y1 = y-sf if y-sf > 0 else 0
        y2 = y+sf if y+sf < img_height else img_height - 1
              
        # Adding the scaling values to the padding list
        x_pad.append(x1)
        x_pad.append(x2)
        
        y_pad.append(y1)
        y_pad.append(y2)
        
    # Converting the min and max value to int
    x_min = int(min(x_pad))
    x_max = int(max(x_pad))
    y_min = int(min(y_pad))
    y_max = int(max(y_pad))
    
    return (x_min, y_min, x_max, y_max)

def point_box(image, x_vals, y_vals, scaling_factor=100):
    """Returns an array of points.
    
    Returns the box that has the length
    of twice the scaling_factor pixels.
    
    Parameters
    ----------
    image: numpy.ndarray
        An RGB image.
    x_vals: list
        A list of x co-ordinates.
    y_vals: list
        A list of y co-ordinates.
    scaling_factor: int
        Scaling the area vertically and horizontally
        for a point.
        
    Returns
    -------
    list
        A list of ``numpy.ndarray`` that consists of the
        end co-ordinates of the box around the pixel.
        
    Examples
    --------
    >>> import numpy as np
    >>> I = np.identity(5)
    >>> x = list(range(1,3))
    >>> y = list(range(1,3))
    >>> boxes = point_box(I, x, y, scaling_factor=100)
    
    """
    boxes = []
    
    for x, y in zip(x_vals, y_vals):
        x_min, y_min, x_max, y_max = region_box_coords(image, [x], [y], scaling_factor=scaling_factor)
        
        polygon = np.array([
            [(x_min, y_min), 
             (x_min, y_max),
             (x_max, y_max),
             (x_max, y_min)
            ]
        ])
        boxes.append(polygon)
        
    return boxes 

def region_of_interest(image, boxes):
    """Returns the image with region of interest.
    
    The image consists of only the region of interest
    with the original image while the rest of the 
    region is colored black with 0.
    
    Parameters
    ----------
    image: numpy.ndarray
        An RGB image.
    boxes: list
        A list of ``numpy.ndarray``.
        
    Returns
    -------
    numpy.ndarray
        An image with the region of interest.
        Use ``matplotlib.pyplot.imshow(image)``
        to see the image in Jupyter Notebook.
    
    Examples
    --------
    >>> import numpy as np
    >>> boxes = [np.array([[1,2],[3,4]]), np.array([[1,2],[3,4]])]
    >>> image = np.identity(3)
    >>> im_roi = region_of_interest(image, boxes)

    """

    mask = np.zeros_like(image)

    for box in boxes:
        image_poly = cv2.fillPoly(mask, box, color=(255,255,255))
        
    masked_image = cv2.bitwise_and(image, image_poly)
    
    return masked_image

def isolate_point_roi(I, boxes):
    """Returns a list of all the roi cropped images around the fingertip.
    
    The cropped images use the point box to isolate the region.
    
    Parameters
    ----------
    I: np.ndarray
        A numpy array.
        This array represents the binary image.
    boxes: list
        A list of all the box co-ordinates in the format used
        for polyfill function.
        **Example**: ``boxes = [np.array([[483,327], [483, 527], [683, 527], [683, 327]])]``
        
        The co-ordinates specified are as ``[[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]]``.
    
    Returns
    -------
    list
        A list tuples of cropped roi images and its top-left position in
        in the input image.
        The first value in the tuple is a numpy array.
        The second value in the tuple is a tuple of ``(ymin, xmin)``
        co-ordinate which corresponds to ``(row, column)``.
    
    Examples
    --------
    >>> I = np.zeros([5,5])
    >>> boxes = [np.array([[[1,1],[1,3],[3,3],[3,1]]])]
    >>> roi_imgs = isolate_point_roi(I, boxes)
    
    """
    roi_images = []

    for box in boxes:
        # Columns
        xmin = box[0][0][0]
        xmax = box[0][2][0]
        
        # Rows
        ymin = box[0][0][1]
        ymax = box[0][2][1]
        
        img_crop = I[ymin:ymax,xmin:xmax]

        roi_images.append((img_crop, (ymin, xmin)))
        
    return roi_images

def roi_edges(roi_crops, blur_n=1, blur_kernel=(5,5), lower_threshold=50, upper_threshold=100, L2gradient=False):
    """Returns a list of tuples of value.
    
    The first value of the tuple is the roi cropped image
    from the main image with edge detected.
    The second value is a tuple with the top-left co-ordinate
    of the roi_cropped image from the main image.
    
    Parameters
    ----------
    roi_crops: list
        A list of tuples. 
        **Example**: ``[(np.array([[1,2],[3,4]]), (0,0)), (np.array([[1,2],[3,4]]), (0,0))]``
    blur_n: int, default ``1``
        The number of times to perform blur operation.
    blur_kernel: tuple, default ``(5,5)``
        The kernel size for blurring operation.
    lower_threshold: int, default ``50``
        The lower threshold value used for Canny edge detection.
    upper_threshold: int, default ``100``
        The upper threshold value used for Canny edge detection.
    L2gradient: bool, default ``False``
        If ``True``, then the L2gradient operation is performed.
        
    Returns
    -------
    list
        A list of tuple of value roi cropped images with edge detected
        and the second value of the top-left co-ordinate in the main image
        from which the roi was cropped.
    
    Examples
    --------
    >>> I_roi0 = np.random.randint(255,size=(100,100), dtype='uint8')
    >>> I_roi1 = np.random.randint(255,size=(100,100), dtype='uint8')
    >>> roi_crops = [(I_roi0, (1,1)), (I_roi1, (2,2))]
    >>> crop_edges = roi_edges(roi_crops)
    
    """
    crop_edges = []
    
    for i in roi_crops:
        img = i[0]
        co = i[1]
        
        # blurring
        for n in range(blur_n):
            img = cv2.blur(img, ksize=blur_kernel)
        
        # Canny edge detection
        edge_img = cv2.Canny(img, threshold1=lower_threshold, threshold2=upper_threshold, L2gradient=L2gradient)
        
        # adding the edge detected image to the crop edge list with top-left co-ordinate
        crop_edges.append((edge_img, co))
        
    return crop_edges

def patch_roi(I, roi_images):
    """Returns the image patched with all the ``roi_images``.
    
    The roi cropped images along with the top-left co-ordinates
    in the main image locations are important.
    
    Parameters
    ----------
    I: np.ndarray
        A numpy array.
        This array represents the binary image.
    roi_images: list
        A list of tuples consisting of numpy array and top-left coordinate
        of the roi image in the main image.
        
    Returns
    -------
    numpy.ndarray
        A patched image with all the roi images.
        
    Examples
    --------
    >>> import numpy as np
    >>> I = np.zeros([5,5])
    >>> I_roi0 = np.array([[1,2],[3,4]])
    >>> I_roi1 = np.array([[1,2],[3,4]])
    >>> roi_I = [(I_roi0, (1,1)), (I_roi1, (2,2))]
    >>> patch_roi(I, roi_I)
    
    """
    # Iterating over all the cropped images
    for i in roi_images:
        # separating the tuple values
        img = i[0]
        coord = i[1]
        
        # Assigning the mininum row coordinate
        row_min = coord[0]
        
        # Reassigning to the variable r
        r = row_min
        
        # Iterating over the crop image row
        for row in range(img.shape[0]):
            # Assigning the column minium coordinate
            col_min = coord[1]
            
            # Reassigning to the variable c
            c = col_min
            
            # Iterating over the crop image columns
            for col in range(img.shape[1]):
                # Assigning the crop image's element to the main image
                I[r][c] = img[row][col]
                
                # Incrementing the column value in the main image
                c+=1
            # Incrementing the row value in the main image
            r+=1
    
    return I

def remove_smaller_clusters(binary_image):
    """
    Identifies clusters in an edge-detected image and removes the smaller one.
    
    Parameters
    -----------    
    binary_image: numpy.ndarray
        Binary edge-detected image.
        
    Returns
    --------
    output_image: numpy.ndarray
        Image containing only the largest cluster.

    Examples
    --------
    >>> edge_image = cv2.imread('~/path/to/file.png', 0)
    >>> binary_image = np.where(edge_image>=1, 1, 0)
    >>> img = remove_smaller_cluster(bin_image.astype(np.uint8))
    
    """
    # connectivity=8 looks at all 8 pixels surrounding a pixel
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    # If there are fewer than 2 clusters (excluding background), return the image as is
    if num_labels <= 2:
        return binary_image

    # Extract areas (stats[label, cv2.CC_STAT_AREA])
    # Label 0 is always the background, so we slice from index 1
    areas = stats[1:, cv2.CC_STAT_AREA]
    
    # Find the index of the largest cluster
    # We add 1 because we sliced the background out of the 'areas' array
    largest_label = np.argmax(areas) + 1

    # Create a mask where only the largest cluster is kept
    # result is 1 where labels == largest_label, else 0
    output_image = np.zeros_like(binary_image)
    output_image[labels == largest_label] = 1

    return output_image.astype(np.uint8)