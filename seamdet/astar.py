from __future__ import annotations

import time
from math import sqrt


# 1 for manhattan, 0 for euclidean
HEURISTIC = 0

delta = [[-1, 0],  # left
    [-1, 1], # left-down
    [0, -1],  # down
    [1, -1],  # right-down
    [1, 0],  # right
    [1, 1],  # right-up
    [0, 1],  # up
    [-1, 1] ] # left-up

TPosition = tuple[int, int]

class Node:
    """Node for A* search algorithm.
    
    """

    def __init__(
        self,
        pos_x: int,
        pos_y: int,
        goal_x: int,
        goal_y: int,
        g_cost: int,
        parent: Node | None,
    ) -> None:
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos = (pos_y, pos_x)
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.g_cost = g_cost
        self.parent = parent
        self.h_cost = self.calculate_heuristic()
        self.f_cost = self.g_cost + self.h_cost

    def calculate_heuristic(self) -> float:
        """
        Heuristic for the A*
        """
        dy = self.pos_x - self.goal_x
        dx = self.pos_y - self.goal_y
        if HEURISTIC == 1:
            return abs(dx) + abs(dy)
        else:
            return sqrt(dy**2 + dx**2)

    def __lt__(self, other: Node) -> bool:
        return self.f_cost < other.f_cost

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self.pos == other.pos

    def __hash__(self) -> int:
        return hash(self.pos)


class AStar:
    """This is Astar search algorithm.

    The algorithm is taken from the _open-source github project.
    The only modification made to this project is the ``delta`` list variable.
    Functionality is added for the algorithm to travel diagonally.


    .. _open-source: https://github.com/TheAlgorithms/Python/blob/master/graphs/bidirectional_a_star.py 

    Parameters
    ----------
    start: tuple
        Coordinates to start the search.
    goal: tuple
        Coordinates to end the search.
    
    Attributes
    ----------
    open_nodes: list
        Stores the open nodes.
    closed_nodes: list
        Stores the closed nodes.
    reached: bool
        Keepds track whether a goal is reached.

    Methods
    -------
    search()
        Performs the search.
    get_successors()
        Returns a list of successors.
    retrace_path()
        Retraces the path from parents to parents until start node.

    Examples
    --------
    >>> grid_shape = (10, 10)
    >>> I = np.ones(grid_shape)
    >>> I[1,1] = 0
    >>> I[2,2] = 0
    >>> I[3,3] = 0
    >>> I[4,4] = 0
    >>> I[5,5] = 0
    >>> I[6,6] = 0
    >>> init = [1,1]
    >>> goal = [6,6]
    >>> grid = copy.copy(I)
    >>> a_star = AStar(init, goal)
    >>> path = a_star.search()
    >>> print(path)
    [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]

    """

    def __init__(self, start: TPosition, goal: TPosition):
        self.start = Node(start[1], start[0], goal[1], goal[0], 0, None)
        self.target = Node(goal[1], goal[0], goal[1], goal[0], 99999, None)

        self.open_nodes = [self.start]
        self.closed_nodes: set[Node] = set()

        self.reached = False

    def search(self) -> list[TPosition]:
        while self.open_nodes:
            # Open Nodes are sorted using __lt__
            self.open_nodes.sort()
            current_node = self.open_nodes.pop(0)

            if current_node.pos == self.target.pos:
                return self.retrace_path(current_node)

            self.closed_nodes.add(current_node)
            successors = self.get_successors(current_node)

            for child_node in successors:
                if child_node in self.closed_nodes:
                    continue

                if child_node not in self.open_nodes:
                    self.open_nodes.append(child_node)
                else:
                    # retrieve the best current path
                    better_node = self.open_nodes.pop(self.open_nodes.index(child_node))

                    if child_node.g_cost < better_node.g_cost:
                        self.open_nodes.append(child_node)
                    else:
                        self.open_nodes.append(better_node)

        return [self.start.pos]

    def get_successors(self, parent: Node) -> list[Node]:
        """
        Returns a list of successors (both in the grid and free spaces)
        """
        successors = []
        for action in delta:
            pos_x = parent.pos_x + action[1]
            pos_y = parent.pos_y + action[0]
            if not (0 <= pos_x <= len(grid[0]) - 1 and 0 <= pos_y <= len(grid) - 1):
                continue

            if grid[pos_y][pos_x] != 0:
                continue

            successors.append(
                Node(
                    pos_x,
                    pos_y,
                    self.target.pos_y,
                    self.target.pos_x,
                    parent.g_cost + 1,
                    parent,
                )
            )
        return successors

    def retrace_path(self, node: Node | None) -> list[TPosition]:
        """
        Retrace the path from parents to parents until start node
        """
        current_node = node
        path = []
        while current_node is not None:
            path.append((current_node.pos_y, current_node.pos_x))
            current_node = current_node.parent
        path.reverse()
        return path