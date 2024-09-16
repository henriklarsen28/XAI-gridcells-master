import sys
from heapq import heapify, heappush, heappop



def manhattanDistance(graph, src, goal):
    """
    Calculates the Manhattan distance between two nodes in a graph.

    Parameters:
    - graph (dict): A dictionary representing the graph.
    - src (int): The source node.
    - goal (int): The goal node.

    Returns:
    - int: The Manhattan distance between the source and goal nodes.
    """
    checkpoint = (3,17)
    checkpoint_dist = abs(graph[src].pos[0] - graph[checkpoint].pos[0]) + abs(graph[src].pos[1] - graph[checkpoint].pos[1])
    return abs(graph[src].pos[0] - graph[goal].pos[0]) + abs(graph[src].pos[1] - graph[goal].pos[1]) + checkpoint_dist

def astar(graph, src, goal):
    """
    A* algorithm implementation for finding the shortest path in a graph.

    Parameters:
    - graph (dict): The graph represented as a dictionary where the keys are the nodes and the values are objects representing the nodes.
    - src: The source node from which to start the search.
    - goal: The goal node to reach.

    Returns:
    - list: The list of nodes representing the shortest path from the source node to the goal node.
    """
    inf = sys.maxsize
    node_data = {}

    #Initialize the list of nodes with fn, cost and predecessor
    for node in graph:
        node_data[node] = {"fn": inf,"cost": inf, "pred": []}

    # Set the source node's cost and fn to 0
    node_data[src]["cost"] = 0
    node_data[src]["fn"] = graph[src].weight

    # Initialize the min_heap with the source node
    min_heap = []
    min_heap.append((graph[src].weight, 0, src))

    # Initialize the visited list to keep track of visited nodes
    visited = []

    while node_data[goal]["cost"] == inf:
        heapify(min_heap)

        # Finds the node with the lowest fn from the min_heap
        f, current_distance, current_node = heappop(min_heap)

        visited.append(current_node)
        for i, neighbor in enumerate(graph[current_node].neighbors):

            # Skip if the neighbor is None
            if graph[current_node].neighbors[i] is None:
                continue
            # Calculate the cost and fn of the neighbor and add them to the heap
            if graph[current_node].neighbors[i].pos not in visited:
                neighbor = graph[current_node].neighbors[i].pos
                cost = node_data[current_node]["cost"] + graph[neighbor].weight

                # Calculate the fn by an average of the current goal position and the end goal position
                fn = cost + manhattanDistance(graph, neighbor, goal)
                # Update the node data
                node_data[neighbor]["cost"] = cost
                node_data[neighbor]["pred"] = node_data[current_node]["pred"] + [current_node]
                node_data[neighbor]["fn"] = fn

                # Add the node to the min_heap
                heappush(min_heap, (node_data[neighbor]["fn"], node_data[neighbor]["cost"], neighbor))
    # Prints the path and cost
    #print(f"Shortest path: {node_data[goal]['pred']}")
    #print(f"Cost: {node_data[goal]['cost']}")
    return node_data[goal]["pred"]