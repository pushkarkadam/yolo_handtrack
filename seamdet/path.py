import numpy as np 
import cv2
import matplotlib.pyplot as plt
import sys
import pickle
from mpl_point_clicker import clicker
import os
import pandas as pd


def get_path_coords(I):
    """Returns the non-black path coordinates.

    Parameters
    ----------
    I: numpy.ndarray
        Binary image of ``m x n`` size.

    Returns
    -------
    list
        A list of coordinates.
        
    """
    # Ensuring the image consists of values 1s and 0s
    I = np.where(I >= 1, 1, 0)
    # import pdb;pdb.set_trace()

    # Identifying non zero element coordinates
    y_co, x_co = np.nonzero(I != 0)

    # Organising the x, y coordinates in a list
    coords = [(x,y) for x, y in zip(x_co, y_co)]

    return coords

def euclidean_distance(p1, p2):
    """Provides Euclidean distance between the two points.

    Paramters
    ---------
    p1: tuple
        Point 1 ``(x, y)`` coordinates.
    p2: tuple
        Point 2 ``(x, y)`` coordinates.

    Returns
    -------
    float
        Euclidean distance between point 1 and 2.
        
    """

    d = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    return d

def get_tracking_endpoints(tracks_df):
    """Returns the tracking endpoints.
    
    Parameters
    ----------
    tracks_df: pandas.dataFrame
        A pandas dataframe with ``timestamp``, ``'x'``, and ``'y'`` values.

    Returns
    -------
    list
        A list of two tuples ``[start, end]``
        
    """
    # Extracting first point
    first_point = tracks_df.iloc[0]
    start = (first_point['x'], first_point['y'])

    # Extracting last point
    last_point = tracks_df.iloc[-1]
    end = (last_point['x'], last_point['y'])

    tracking_ends = [start, end]

    return tracking_ends

def get_end_nodes(I, coords):
    """Provides end nodes.

    Parameters
    ----------
    I: numpy.ndarray
        Binary image of ``m x n`` size.
    coords: list
        A list of tuples of ``(x, y)`` coordinates.

    Returns
    -------
    list
        A list of tuple of end nodes.
    
    """
    end_nodes = []

    for coord in coords:
        c, r = coord
        N = I[r-1:r+2, c-1:c+2]
        if np.sum(N) == 2:
            end_nodes.append(coord)    

    return end_nodes

def true_end_nodes(end_nodes, tracking_ends):
    """Selects the end nodes based on the close proximity to the tracking end nodes.

    Parameters
    ----------
    end_nodes: list
        A list of tuple from the possible detected end nodes.
    tracking_ends: list
        A list of tuple that indicate a rough estimation of where the node end points lie.

    Returns
    -------
    tuple
        Start and goal node coordinates in ``(x, y)`` tuple.
    
    """
    start_d = []
    goal_d = []

    for node in end_nodes:
        sd = euclidean_distance(node, tracking_ends[0])
        gd = euclidean_distance(node, tracking_ends[1])

        start_d.append(sd)
        goal_d.append(gd)

    start = [node for _, node in sorted(zip(start_d, end_nodes))]
    goal = [node for _, node in sorted(zip(goal_d, end_nodes))]

    start_node = start[0]
    goal_node = goal[0]

    return start_node, goal_node

class Node:
    """
    Node class that keeps track of the path.

    Parameters
    ----------
    position: tuple
        Position of the path in terms of x and y coordinates.
    parent: Node, default ``None``
        A node linking the current node to its parent.

    """
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent

def search_path(start_position, goal_position, I):
    """Develops are search path.
    
    Parameters
    ----------
    start_position: tuple
        The start coordinate of the segment.
    goal_position: tuple
        The goal coordinate of the segment.
    I: numpy.ndarray
        A numpy matrix of ``m x n`` size.
    
    Returns
    -------
    list 
        A list of Node objects.
        
    """

    start_node = Node(position=start_position)
    goal_node = Node(position=goal_position)

    # Keeping track of node visisted
    node_visited = [start_node.position]

    # Starting path list
    path = [start_node]

    # Assigning start_node to node to begin
    node = start_node

    # Counter
    count = 0

    while node.position != goal_position and count < 999:
        c, r = node.position
        N = I[r-1:r+2, c-1:c+2]
        rows, cols = np.nonzero(N != 0)

        # Adjusting the index to the image
        rows = [r - 1 + rx for rx in rows]
        cols = [c - 1 + cx for cx in cols]

        # Gets local path in the search kernel
        local_path = [(x,y) for  y, x in zip(rows, cols)]

        # Sort the coordinates by their euclidean distance to the goal
        local_path = sort_points_by_distance(local_path, goal_position)
        local_path_set = set(local_path)

        # Checks if the path has been visited
        # This is implemented to avoid sticking in local regions to backtrack to parent
        if local_path_set.issubset(set(node_visited)):
            # Removing the node from path where the algorithm reaches dead-end
            path.pop(-1)
            
            # Using previous node 
            node = path[-1]

            # Continuing to next iteration
            continue

        for cx, cy in local_path:
            if (cx, cy) not in node_visited:
                next_node = Node(position=(cx, cy))
                next_node.parent = node
                node_visited.append((cx, cy))
                path.append(next_node)

        node = next_node

        count+=1
        
    
    return path

def find_goal_in_path(goal_position, path):
    """Finds the goal position in path.
    
    Paramters
    ---------
    goal_position: tuple
        Coordinates of goal position.
    path: list
        A list of path nodes.

    Returns
    -------
    int
        Index of the goal node in the path.
        
    """

    for idx, node in enumerate(path):
        if node.position == goal_position:
            return idx

def get_path_ancestry(path, start_position, goal_position):
    """Returns path from the end node to start node.
    
    Parameters
    ----------
    path: list
        A list of Node objects indicating path.
    start_position: tuple
        Coordinates of start position of the segment.
    goal_position: tuple
        Coordinates of goal position of the segment.

    Returns
    -------
    list
        A list of tuples organised from start to the end coordinates of the segment.
    """

    goal_idx = find_goal_in_path(goal_position, path)

    goal_node = path[goal_idx]

    prev_node = goal_node

    path_coords = []

    while True:
        # Extracting position 
        x, y = prev_node.position

        # Appending the position
        path_coords.append((x, y))

        prev_node = prev_node.parent

        if not prev_node:
            break

    path_coords.reverse()

    return path_coords

def sort_points_by_distance(coords, goal):
    """Sorts the coordinates in descending order of their euclidean
    distance from the goal.

    The descending order approach is selected such that the node with the smallest
    distance (i.e. one closer to the goal) is added last to the path.
    This ensures that the node closest to the goal is selected during backtracking.

    Parameters
    ----------
    coords: list
        A list of tuple of the neighbouring points detected.
    goal: tuple
        Goal coordinate tuple ``(x, y)``.

    Returns
    -------
    list
        A list of neighbour coordinates organised in descending order of their distance
        from the goal.
    
    """

    distances = [] 

    for c in coords:
        d = euclidean_distance(c, goal)
        distances.append(d)

    coords_sorted = [coord for _, coord in sorted(zip(distances, coords))]

    return list(reversed(coords_sorted))

def combine_images(images):
    """Combines the images.

    Parameters
    ----------
    images: list
        A list of numpy.ndarray images.

    Returns
    -------
    numpy.ndarray
        A combined image of all the images.
        
    """

    composite = np.sum(images, axis=0)

    return composite

def get_composite_paths(images, tracking_endpoints):
    """Performs composite seam line detection.
    
    Paramters
    ---------
    images: list
        A list of numpy.ndarray images.
    tracking_endpoints: dict
        A dictionary of tracking end points used to find the true start and goal position.
        Example: ``{0: [(50, 50), (100, 100)], 1: [(20, 20), (30, 20)]}``

    Return
    ------
    dict
        A dictionary of composite paths.
        
    """

    composite_paths = dict()

    for idx, image in enumerate(images):
        # thinning
        I = cv2.ximgproc.thinning(image.astype(np.uint8))

        # converting to binary image
        I = np.where(I > 1, 1, 0)

        # extracting non-zero coordinates
        coords = get_path_coords(I)

        # Getting all the possible end nodes
        end_nodes = get_end_nodes(I, coords)

        # Extracting the endpoints for the segment
        tracking_ends = tracking_endpoints[idx]

        # Finding true end nodes
        start, goal = true_end_nodes(end_nodes, tracking_ends)

        # Finding search path
        path = search_path(start, goal, I)

        # Finding path ancestry
        path_ancestry = get_path_ancestry(path, start, goal)

        # Creating a dictionary to store path
        composite_paths[idx] = dict()

        # Extracting the tuple to x and y list.
        xl, yl = zip(*path_ancestry)

        # Adding the list to the composite path
        composite_paths[idx]['x'] = xl
        composite_paths[idx]['y'] = yl

    return composite_paths

def save_composite_plot(composite_image, composite_paths, save_path='./composite.png'):
    """Saves the composite image plot.

    Parameters
    ----------
    composite_image: numpy.ndarray
        A numpy image.
    composite_paths: dict
        A dictionary of paths for all the line segments.
    save_path: str, default ``'./composite.png'``
        Path to save the plot.
        
    """
    plt.imshow(composite_image, cmap="gray")

    for k, v in composite_paths.items():
        plt.plot(v['x'], v['y'])

    plt.savefig(save_path)

def checkpoint_bounding_box(checkpoints, size=1):
    """Generates checkpoint bounding box.

    Parameters
    ----------
    checkpoints: list
        A list of checkpoint tuples as ``(x, y)``.
    size: int
        The size of the bounding box in 2x in x and y direction.

    Returns
    -------
    list
        A list of box corner coordinates.
    
    """

    boxes = []

    for x,y in checkpoints:
        x_min = x - size
        x_max = x + size
        y_min = y - size
        y_max = y + size

        box = [x_min, y_min, x_max, y_max]

        boxes.append(box)

    return boxes

def checkpoint_pass_check(path_coord, box):
    """Checks if the point is in the box.

    Parameters
    ----------
    path_coord: tuple
        A tuple of x and y.
    box: list
        A list of top-left and bottom right box coordinates ``[x_min, y_min, x_max, y_max]`` in image plane.

    Returns
    -------
    bool
        A boolean whether the coordinate lies within the bounding box.
        
    """

    x, y = path_coord
    x_min, y_min, x_max, y_max = box

    if (x > x_min and x < x_max) and (y > y_min and y < y_max):
        return True
    
    return False

def evaluate_checkpoints(checkpoints, path, bbox_size=1):
    """Checks the checkpoints visited.

    Parameters
    ----------
    checkpoints: list
        A list of tuple for the checkpoints.
    path: dict
        A dictionary of ``'x'`` and ``'y'`` list.
    bbox_size: int
        Size of the box from the centre point.

    Returns
    -------
    visited_checkpoint: list
        A list of tuples.
    passing_ratio: float
        A ratio of checkpoints completed to the total checkpoints.
        
    """
    path_x, path_y = list(path['x']), list(path['y'])

    checkpoint_num = len(checkpoints)

    visited_checkpoints = []

    bboxes = checkpoint_bounding_box(checkpoints, size=bbox_size)

    for x, y in zip(path_x, path_y):
        for box, checkpoint in zip(bboxes, checkpoints):
            if checkpoint_pass_check((x,y), box):
                visited_checkpoints.append(checkpoint)
                checkpoint_idx = checkpoints.index(checkpoint)
                bboxes.pop(checkpoint_idx)
                checkpoints.pop(checkpoint_idx)
                break

    visited_num = len(visited_checkpoints)
    
    passing_ratio = visited_num / checkpoint_num

    return visited_checkpoints, passing_ratio

def point_plotter(image_path, save_file='', point_labels=["check_points"], marker_style=["x"], colors=['r']):
    """Plots the points on the image by clicking on them.
    This will be done on the rectified image.
    This is helpful for developing ground truth.

    Parameters
    ----------
    image_path: str
        A string of path where the image is stored.
    save_file: str, default ``''``
        A path where the results will be stored. 
        The results are stored as ``.pkl``.
        Provide name without the extension.
        Example: ``'~/path/to/file'``. The code will add ``'~/path/to/file.pkl'`` before saving.
    point_label: list, default ``['check_points']``
        A list of different types of labels to store.
    marker_style: list, default ``["x"]``
        A list of markers to be plotted.
    colors: list, default ``['r']``
        A list of colors for all the markers.

    Returns
    -------
    dict
        A dictionary of data points with x and y list coordinates.

    Examples
    --------
    >>> data = point_plotter('rectified_imageL.png', 'checkpoints', ["check_points", "box"], ["x", "o"], ["r", "b"])
        
    """

    try:
        assert(len(point_labels) == len(marker_style) == len(colors))
    except Exception as e:
        print('\033[93m' + 'The size of the list of point_labels, marker_style, and colors must be equal.')
        sys.exit(1)

    image = cv2.imread(image_path)

    fig, ax = plt.subplots(constrained_layout=True)

    ax.imshow(image)

    klicker = clicker(
        ax,
        point_labels,
        markers=marker_style,
        linestyle="--",
        colors=colors
    )

    plt.show()

    coords = klicker.get_positions()

    # Dictionary to store the data
    data = dict()

    # Iterating over every type of label
    for point_label in point_labels:
        x_list, y_list = zip(*coords[point_label])

        label_data = {'x': list(x_list), 'y': list(y_list)}

        data[point_label] = label_data
    
    if save_file:
        with open(save_file + '.pkl', 'wb') as f:
            pickle.dump(data, f)
            
    return data

def load_checkpoint_gt(path):
    """Loads the checkpoint ground truth data.
    
    Parameters
    ----------
    path: str
        Path to the pickle file.
        Example: ``'~/path/to/pickle_file.pkl'``

    Returns
    -------
    dict
        A dictionary of data stored.

    Examples
    --------
    >>> data = load_checkpoint_gt('checkpoints.pkl')
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data

def get_robot_coords(hand_eye_calibration_path, camera_coords_df, save_path):
    """Returns the robot coordinates.
    
    Parameters
    ----------
    hand_eye_calibration_path: str
        Path to the directory where the calibration data is stored.
    camera_coords_df: pandas.DataFrame
        A pandas dataframe with camera coordinates.
    save_path: str
        Path to the directory to store the data

    Returns
    -------
    pandas.DataFrame
        A pandas dataframe with the robot coordinates.
        
    """

    rtc_path = rTc_path = os.path.join(hand_eye_calibration_path, 'rTc.npz')

    # loading robot hand eye calibration data
    rTc_data = np.load(rTc_path)

    rTc = rTc_data['rTc']

    column_names = ['x', 'y', 'z']

    # Extracting the x, y, z coordinates from camera coordinates dataframe
    xc, yc, zc = [np.array(camera_coords_df[i]) for i in column_names]

    ones = np.ones(xc.shape)

    # Creating a matrix of camera x, y, z values
    X_c = np.vstack([xc, yc, zc, ones])

    # Transforming by multiplying camera to robot Transformation matrix
    X_r = np.matmul(rTc, X_c)

    # Creating data frame by first transposing the X_r and then selecting all rows and the first three columns
    # of the newly transformed matrix.
    robot_coords_df = pd.DataFrame(X_r.T[:, :-1], columns=column_names)

    if save_path:
        robot_coords_df.to_csv(save_path, index=False)

    return robot_coords_df