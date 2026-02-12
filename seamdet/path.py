import numpy as np 
import cv2


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
        local_path = set([(x,y) for  y, x in zip(rows, cols)])

        # Checks if the path has been visited
        # This is implemented to avoid sticking in local regions to backtrack to parent
        if local_path.issubset(set(node_visited)):
            # Removing the node from path where the algorithm reaches dead-end
            path.pop(-1)
            
            # Using previous node 
            node = path[-1]

            # Continuing to next iteration
            continue

        for cx, cy in zip(cols, rows):
            if (cx, cy) not in node_visited:
                next_node = Node(position=(cx, cy))
                next_node.parent = node
                node_visited.append((cx, cy))
                path.append(next_node)

        node = next_node

        print(f'count: {count}')
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