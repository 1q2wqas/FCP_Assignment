import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
from matplotlib.animation import FuncAnimation


class Node:

    def __init__(self, value, number, connections=None):
        self.index = number
        self.connections = connections
        self.value = value


class Network:

    def __init__(self, nodes=None):

        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes

    def add_edge(self, i, j):
        self.nodes[i].connections.append(j)
        self.nodes[j].connections.append(i)


    def get_mean_degree(self):
        """
        Calculate and return the average degree of the nodes in the network.
        The degree of a node is simply the number of edges it has.
        """
        # Initialize the total degrees to 0.
        degrees = 0
        for node in self.nodes:
            # Determine how many "1s" there are in the connections list—these numbers represent a connection.
            degrees += node.connections.count(1)

        if len(self.nodes) > 0:
            # If there are nodes in the network, calculate the sum of the degrees of all nodes divided by the total number of nodes to get the average degree.
            mean_degrees = degrees / len(self.nodes)
        else:
            # If there are no nodes in the network, the average degree is set to 0.
            mean_degrees = 0

        # Returns the calculated average degree.
        return mean_degrees

    def get_clustering(self):
        """
        Determine the network's average clustering coefficient and return the output.
        The ratio of actual edges between a node's neighbors to the total number of
        potential edges between them is known as the clustering coefficient.
        """

        if len(self.nodes) == 0:
            # If there are no nodes in the network, return a clustering coefficient of 0,
            return 0

        # Initialize the total clustering coefficient to 0.
        total_clustering = 0
        for node in self.nodes:
            if len(node.connections) < 2:
                # If a node has less than 2 neighbours, the clustering coefficient is 0 as triangles cannot be formed
                continue
            # Build a list to store the indices of all neighbors directly connected to this node.1 means a collection, 0 means no collection.

            neighbors = []
            # Loop through the connected array of the current node
            for neighbor_index, connected in enumerate(node.connections):
                if connected == 1:
                    # Add the index of this neighbour to the neighbour list.
                    neighbors.append(neighbor_index)

            # Initialize the count of existing connections to 0.
            links = 0
            # Calculate the maximum number of possible connections.
            possible_links = len(neighbors) * (len(neighbors) - 1) / 2

            # Iterate over all combinations of neighbours.
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    # If there is a direct connection between neighbor i and neighbor j.
                    if self.nodes[neighbors[i]].connections[neighbors[j]] == 1:
                        # links increment
                        links += 1

            # Calculate the clustering coefficient for this node.
            if possible_links > 0:
                # The clustering coefficient is the ratio of links to possible links.
                node_clustering = links / possible_links
            else:
                # If there are no possible links，the clustering coefficient is 0.
                node_clustering = 0
            # Accumulate the clustering coefficients of all nodes.
            total_clustering += node_clustering

        # Calculate and return the average clustering coefficient for all nodes
        return total_clustering / len(self.nodes)

    def get_path_length(self):
        """
        Determine and return the average shortest path length between all pairs of nodes in the network.
        """

        # Initialize the total path length to 0.
        total_path_length = 0
        # Count the total number of nodes in the network.
        num_nodes = len(self.nodes)

        for start_node in self.nodes:
            # Initialize distances from the start_node to all others as infinite.
            distances = {node.index: np.inf for node in self.nodes}
            # Set the distance from the start_node to itself as zero.
            distances[start_node.index] = 0
            # Initialize the queue for the Breadth-First Search with the start node and its distance.
            queue = [(start_node, 0)]
            # loop queue to find the shortest paths from the start_node.
            while queue:
                # Pop the first node in the queue.
                current_node, current_distance = queue.pop(0)
                # Loop over all possible neighbors of the current node.
                for neighbor_index, is_connected in enumerate(current_node.connections):
                    # Verify whether a link exists and whether the neighbour got any visits.
                    if is_connected == 1 and distances[neighbor_index] == np.inf:
                        # Increase the neighbour's distance and submit it for further research.
                        distances[neighbor_index] = current_distance + 1
                        queue.append((self.nodes[neighbor_index], current_distance + 1))

            # sum up the distances from the start_node to all other nodes.
            total_path_length += sum(distances.values())

        # Calculate the average of the shortest paths between all pairs of nodes.
        if num_nodes > 1:
            # Divide the whole path length by the total number of node pairs that are possible.
            mean_path_length = total_path_length / (num_nodes * (num_nodes - 1))
        else:
            # If there's only one node, the mean path length is zero.
            mean_path_length = 0
        # Round the result to 15 decimal places and return the result.
        return round(mean_path_length,15)

    def make_random_network(self, N, connection_probability=0.5):
        '''
		This function makes a *random* network of size N.
		Each node is connected to each other node with probability p
		'''

        self.nodes = []
        for node_number in range(N):
            value = np.random.random()
            connections = [0 for _ in range(N)]
            self.nodes.append(Node(value, node_number, connections))

        for (index, node) in enumerate(self.nodes):
            for neighbour_index in range(index + 1, N):
                if np.random.random() < connection_probability:
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1

    def make_ring_network(self, N, neighbour_range=2):
        self.nodes = []
        for i in range(N):
            connections = []
            for j in range(1, neighbour_range + 1):
                prev_index = (i - j) % N  # 使用模运算确保索引有效
                next_index = (i + j) % N  # 使用模运算确保索引有效
                connections.extend([prev_index, next_index])
            self.nodes.append(Node(value=i, number=i, connections=connections))

    def make_small_world_network(self, N, re_wire_prob=0.2):
        self.make_ring_network(N)  # 首先创建一个环形网络
        for node in self.nodes:
            new_connections = []
            for neighbor_index in node.connections:
                if np.random.rand() < re_wire_prob and len(self.nodes) > 1:
                    # 选择一个除当前节点和已有连接外的随机节点
                    choices = [n for n in range(N) if n != node.index and n not in node.connections]
                    if choices:
                        new_neighbor = np.random.choice(choices)
                        new_connections.append(new_neighbor)
                    else:
                        new_connections.append(neighbor_index)
                else:
                    new_connections.append(neighbor_index)
            node.connections = new_connections

    def plot(self, network_type=None, re_wire_prob=None):
        """
		This function plots the network.

		Arguments:
			network_type (str): Type of network (e.g., "Ring", "Small-World").
			re_wire_prob (float): Rewiring probability for small-world networks.
		"""

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        num_nodes = len(self.nodes)
        network_radius = num_nodes * 10
        ax.set_xlim([-1.1 * network_radius, 1.1 * network_radius])
        ax.set_ylim([-1.1 * network_radius, 1.1 * network_radius])

        for (i, node) in enumerate(self.nodes):
            node_angle = i * 2 * np.pi / num_nodes
            node_x = network_radius * np.cos(node_angle)
            node_y = network_radius * np.sin(node_angle)

            circle = plt.Circle((node_x, node_y), 0.3 * num_nodes, color=cm.hot(node.value))
            ax.add_patch(circle)

            for neighbour_index in node.connections:
                neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                neighbour_x = network_radius * np.cos(neighbour_angle)
                neighbour_y = network_radius * np.sin(neighbour_angle)

                ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')

        # Add title indicating network type and rewiring probability (if applicable)
        if network_type:
            if re_wire_prob is not None:
                title = network_type + " Network (Re-wiring Probability = " + str(re_wire_prob) + ")"
            else:
                title = network_type + " Network (Range 2)"
            plt.title(title)

        plt.show()


def test_networks():
    # Ring network
    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number - 1) % num_nodes] = 1
        connections[(node_number + 1) % num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing ring network")
    assert (network.get_mean_degree() == 2), network.get_mean_degree()
    assert (network.get_clustering() == 0), network.get_clustering()
    assert (network.get_path_length() == 2.777777777777778), network.get_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number + 1) % num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing one-sided network")
    assert (network.get_mean_degree() == 1), network.get_mean_degree()
    assert (network.get_clustering() == 0), network.get_clustering()
    assert (network.get_path_length() == 5), network.get_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [1 for val in range(num_nodes)]
        connections[node_number] = 0
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing fully connected network")
    assert (network.get_mean_degree() == num_nodes - 1), network.get_mean_degree()
    assert (network.get_clustering() == 1), network.get_clustering()
    assert (network.get_path_length() == 1), network.get_path_length()

    print("All tests passed")


'''
==============================================================================================================
This section contains code for the Ising Model - task 1 in the assignment
==============================================================================================================
'''


def calculate_agreement(population, row, col, external=0.0):
    """Calculate agreement of a given cell with its direct neighbors."""
    n_rows, n_cols = population.shape
    # Define neighbor indices using periodic boundary conditions
    neighbor_indices = [
        ((row - 1) % n_rows, col),  # top
        ((row + 1) % n_rows, col),  # bottom
        (row, (col - 1) % n_cols),  # left
        (row, (col + 1) % n_cols)  # right
    ]
    # Gather neighbor values
    neighbors = [population[i] for i in neighbor_indices]
    # Calculate agreement score including external influence
    agreement = sum(neighbors) * population[row, col] + external * population[row, col]
    return agreement


def ising_step(population, beta, external=0.0):
    """Perform a single update step in the Ising model."""
    n_rows, n_cols = population.shape
    # Perform updates for each cell in the grid
    for _ in range(n_rows * n_cols):
        row = np.random.randint(0, n_rows)
        col = np.random.randint(0, n_cols)
        # Calculate energy change if cell state is flipped
        delta_e = calculate_agreement(population, row, col, external)
        # Flip the cell with a probability dependent on delta_e and beta
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e * beta):
            population[row, col] *= -1


def plot_ising(population):
    """Display the Ising model using Matplotlib."""
    plt.imshow(population, interpolation='none', cmap='RdPu_r')
    plt.axis('off')
    plt.draw()


def simulate_ising(model, steps=1000, frames=100, pause_time=0.01):
    """Run the Ising model simulation."""
    plt.ion()  # Turn on interactive mode for live updates
    for frame in range(frames):
        # Update the model state multiple times before redrawing
        for step in range(steps):
            ising_step(model['population'], model['beta'], model['external'])
        plot_ising(model['population'])
        plt.pause(pause_time)  # Pause to update the display
    plt.ioff()  # Turn off interactive mode
    plt.show()


def test_ising():
    """This function will test the calculate_agreement function in the Ising model."""
    print("Testing ising model calculations")
    population = -np.ones((3, 3))
    assert (calculate_agreement(population, 1, 1) == 4), "Test 1"

    population[1, 1] = 1.
    assert (calculate_agreement(population, 1, 1) == -4), "Test 2"

    population[0, 1] = 1.
    assert (calculate_agreement(population, 1, 1) == -2), "Test 3"

    population[1, 0] = 1.
    assert (calculate_agreement(population, 1, 1) == 0), "Test 4"

    population[2, 1] = 1.
    assert (calculate_agreement(population, 1, 1) == 2), "Test 5"

    population[1, 2] = 1.
    assert (calculate_agreement(population, 1, 1) == 4), "Test 6"

    print("Testing external pull")
    population = -np.ones((3, 3))
    assert (calculate_agreement(population, 1, 1, 1) == 3), "Test 7"
    assert (calculate_agreement(population, 1, 1, -1) == 5), "Test 8"
    assert (calculate_agreement(population, 1, 1, -10) == 14), "Test 9"
    assert (calculate_agreement(population, 1, 1, 10) == -6), "Test 10"

    print("Tests passed")


def ising_main():
    """Main function to handle command-line arguments and launch the simulation or testing."""
    parser = argparse.ArgumentParser(description="Run the Ising model simulation.")
    parser.add_argument("-ising_model", action="store_true", help="Run the Ising model with default parameters")
    parser.add_argument("-external", type=float, default=0.0, help="External influence factor")
    parser.add_argument("-alpha", type=float, default=1.0, help="Alpha value for societal tolerance")
    parser.add_argument("-test_ising", action="store_true", help="Run the test functions for the model")
    parser.add_argument('-network', metavar='N', type=int, help='Random network to create and plot')
    parser.add_argument('-test_network', action='store_true', help='Run network tests')
    args = parser.parse_args()

    if args.alpha == 0:
        print("Error: Alpha value cannot be zero.")
        return  # Early exit from the function

    if args.test_ising:
        test_ising()  # Run tests for the Ising model calculations
    elif args.ising_model:
        population = np.random.choice([-1, 1], size=(30, 30))
        model = {
            'population': population,
            'beta': 1 / args.alpha,
            'external': args.external
        }
        simulate_ising(model)  # Start the Ising model simulation


def simulate_ising_network(network, beta, steps, frames):
    fig, ax = plt.subplots()
    positions = np.column_stack((np.cos(np.linspace(0, 2 * np.pi, len(network.nodes))),
                                 np.sin(np.linspace(0, 2 * np.pi, len(network.nodes)))))

    mean_opinions = []

    def update(frame):
        update_ising_network(network, beta)
        current_mean_opinion = np.mean([node.value for node in network.nodes])
        mean_opinions.append(current_mean_opinion)

        ax.clear()
        ax.set_axis_off()
        # Draw the edges
        for i in range(len(network.nodes)):
            for j in network.nodes[i].connections:
                ax.plot([positions[i][0], positions[j][0]], [positions[i][1], positions[j][1]], 'k-')
        # Draw the nodes
        node_colors = [node.value for node in network.nodes]
        ax.scatter(positions[:, 0], positions[:, 1], c=node_colors, cmap='coolwarm', vmin=-1, vmax=1)

    ani = FuncAnimation(fig, update, frames=frames, repeat=False)
    plt.show()

    # Plot mean opinion over time after the animation
    plt.figure()
    plt.plot(mean_opinions)
    plt.xlabel('Time')
    plt.ylabel('Mean Opinion')
    plt.title('Mean Opinion Over Time')
    plt.show()

def update_ising_network(network, beta):
    for node in network.nodes:
        neighbors = [network.nodes[i] for i in node.connections]
        local_field = sum(neighbor.value for neighbor in neighbors)
        prob = 1 / (1 + np.exp(-beta * local_field * node.value))
        node.value = 1 if np.random.rand() < prob else -1

# Example usage:








'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
==============================================================================================================
'''


def defuant_main():
    # Your code for task 2 goes here
    pass


def test_defuant():
    # Your code for task 2 goes here
    pass


'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''



def main():
    parser = argparse.ArgumentParser(description="网络Ising模拟。")
    parser.add_argument("-ising_model", action="store_true", help="运行Ising模型")
    parser.add_argument("-use_network", type=int, help="使用指定节点数的网络", nargs='?', const=10, default=None)
    parser.add_argument("-alpha", type=float, default=1.0, help="Ising")
    parser.add_argument("-external", type=float, default=0.0, help="External influence factor")
    parser.add_argument("-test_ising", action="store_true", help="Run the test functions for the model")
    parser.add_argument('-ring_network', type=int, help='Specify the size of the ring network')
    parser.add_argument('-small_world', type=int, help='Specify the size of the small-world network')
    parser.add_argument('-re_wire', type=float, default=0.2, help='Specify the rewiring probability (default is 0.2)')
    parser.add_argument('-network', type=int, help='Specify the size of the random network')
    parser.add_argument('-test_network', action='store_true', help='Run the test functions that have provided')

    args = parser.parse_args()

    network = Network()

    if args.ising_model:
        if args.use_network is not None:
            network = Network()
            # 默认使用小世界网络，除非另有指定
            network.make_small_world_network(args.use_network, 0.2)
            simulate_ising_network(network, 1/args.alpha, 1000, 200)

        else:
            # if args.external is not None:
            population = np.random.choice([-1, 1], size=(30, 30))
            model = {
                'population': population,
                'beta': 1 / args.alpha,
                'external': args.external
            }
            simulate_ising(model)
    elif args.test_ising:
        print('Run tests for the Ising model calculations')
        test_ising()  # Run tests for the Ising model calculations

    elif args.ring_network:
        # Create and plot a ring network
        N = args.ring_network
        network.make_ring_network(N)
        network.plot(network_type="Ring")

    elif args.small_world:
        # Create and plot a small-world network
        N = args.small_world
        re_wire_prob = args.re_wire
        network.make_small_world_network(N, re_wire_prob)
        network.plot(network_type="Small-World", re_wire_prob=re_wire_prob)

    elif args.network:
        # Create and plot a random network
        N=args.network
        network.make_random_network(N)
        print("Mean degree:",network.get_mean_degree())
        print("Average path length:",network.get_path_length())
        print("Clustering co-efficient:",network.get_clustering())
        network.plot(network_type="random")

    elif args.test_network:
        # test some different networks
        test_networks()

    else:
        print('please')

if __name__ == "__main__":
    main()