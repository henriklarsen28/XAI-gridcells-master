class Graph:
    class Node:
        def __init__(self, pos, weight=1):
            self.pos = pos
            self.weight = weight
            self.neighbors = [None, None, None, None]  # Up, Right, Down, Left

    def __init__(self, env_map):
        self.env_map = env_map
        self.nodes = []
        self.node_count = 0

    # Check +- 1 in x and y direction
    def add_neighbors(self, node):
        size_of_env = self.env_map.shape
        # Check if the node is on the edge of the map to the left
        if node.pos[0] != 0:
            # Check if the node is a wall
            next_node = int(self.env_map[node.pos[0] - 1][node.pos[1]])
            if next_node == 1:
                pass
            else:
                # Create new node to the left
                node_left = self.Node((node.pos[0] - 1, node.pos[1]), 1)
                node.neighbors[3] = node_left

        # Check if the node is on the edge of the map to the right
        if node.pos[0] != size_of_env[0] - 1:
            # Check if the node is a wall
            next_node = int(self.env_map[node.pos[0] + 1][node.pos[1]])
            if next_node == 1:
                pass
            else:
                # Create new node to the right
                node_right = self.Node((node.pos[0] + 1, node.pos[1]), 1)
                node.neighbors[1] = node_right

        # Check if the node is on the edge of the map upwards
        if node.pos[1] != 0:
            # Check if the node is a wall
            next_node = int(self.env_map[node.pos[0]][node.pos[1] - 1])
            if next_node == 1:
                pass
            else:
                # Create new node upwards
                node_up = self.Node((node.pos[0], node.pos[1] - 1), 1)
                node.neighbors[0] = node_up

        # Check if the node is on the edge of the map downwards
        if node.pos[1] != size_of_env[1] - 1:
            # Check if the node is a wall
            next_node = int(self.env_map[node.pos[0]][node.pos[1] + 1])
            if next_node == 1:
                pass
            else:
                # Create new node downwards
                node_down = self.Node((node.pos[0], node.pos[1] + 1), 1)
                node.neighbors[2] = node_down

    def make_graph(self):

        # Iterate through the map
        for y in range(self.env_map.shape[1]):

            current = False
            nxt = self.env_map[0][y] == 0 or self.env_map[0][y] == 2

            # Buffer for the left node to complete the connection
            for x in range(self.env_map.shape[0]):

                # Update the previous, current and nxt variables
                current = nxt
                try:
                    nxt = self.env_map[x + 1][y] == 0 or self.env_map[x + 1][y] == 2
                except:
                    nxt = False


                # Hit a wall, skip
                if current == False:
                    continue

                node = self.Node((x, y), 1)
                # Check the neighbors and add them as neighbors
                self.add_neighbors(node)
                self.nodes.append(node)

                self.node_count += 1

        # Convert from list to dict
        self.nodes = {node.pos: node for node in self.nodes}

        return self.nodes

    # Print the graph for debugging
    def print_graph(self):
        for node in self.nodes:
            txt = f"Node: {node}, Neighbors: {self.nodes[node].neighbors}"
            print(txt)
