import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
import argparse

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

	def get_mean_degree(self):
		#Your code  for task 3 goes here

	def get_mean_clustering(self):
		#Your code for task 3 goes here

	def get_mean_path_length(self):
		#Your code for task 3 goes here

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
			for neighbour_index in range(index+1, N):
				if np.random.random() < connection_probability:
					node.connections[neighbour_index] = 1
					self.nodes[neighbour_index].connections[index] = 1

	def make_ring_network(self, N, neighbour_range=1):
		#Your code  for task 4 goes here

	def make_small_world_network(self, N, re_wire_prob=0.2):
		#Your code for task 4 goes here

	def plot(self):

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_axis_off()

		num_nodes = len(self.nodes)
		network_radius = num_nodes * 10
		ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
		ax.set_ylim([-1.1*network_radius, 1.1*network_radius])

		for (i, node) in enumerate(self.nodes):
			node_angle = i * 2 * np.pi / num_nodes
			node_x = network_radius * np.cos(node_angle)
			node_y = network_radius * np.sin(node_angle)

			circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color=cm.hot(node.value))
			ax.add_patch(circle)

			for neighbour_index in range(i+1, num_nodes):
				if node.connections[neighbour_index]:
					neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
					neighbour_x = network_radius * np.cos(neighbour_angle)
					neighbour_y = network_radius * np.sin(neighbour_angle)

					ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')

def test_networks():

	#Ring network
	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number-1)%num_nodes] = 1
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing ring network")
	assert(network.get_mean_degree()==2), network.get_mean_degree()
	assert(network.get_clustering()==0), network.get_clustering()
	assert(network.get_path_length()==2.777777777777778), network.get_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing one-sided network")
	assert(network.get_mean_degree()==1), network.get_mean_degree()
	assert(network.get_clustering()==0),  network.get_clustering()
	assert(network.get_path_length()==5), network.get_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [1 for val in range(num_nodes)]
		connections[node_number] = 0
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing fully connected network")
	assert(network.get_mean_degree()==num_nodes-1), network.get_mean_degree()
	assert(network.get_clustering()==1),  network.get_clustering()
	assert(network.get_path_length()==1), network.get_path_length()

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
        ((row - 1) % n_rows, col),   # top
        ((row + 1) % n_rows, col),   # bottom
        (row, (col - 1) % n_cols),   # left
        (row, (col + 1) % n_cols)    # right
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
    plt.ioff()    # Turn off interactive mode
    plt.show()

def test_ising():
    '''
    This function will test the calculate_agreement function in the Ising model.
    '''
    print("Testing ising model calculations")
    population = -np.ones((3, 3))
    assert(calculate_agreement(population, 1, 1) == 4), "Test 1"

    population[1, 1] = 1.
    assert(calculate_agreement(population, 1, 1) == -4), "Test 2"

    population[0, 1] = 1.
    assert(calculate_agreement(population, 1, 1) == -2), "Test 3"

    population[1, 0] = 1.
    assert(calculate_agreement(population, 1, 1) == 0), "Test 4"

    population[2, 1] = 1.
    assert(calculate_agreement(population, 1, 1) == 2), "Test 5"

    population[1, 2] = 1.
    assert(calculate_agreement(population, 1, 1) == 4), "Test 6"

    print("Testing external pull")
    population = -np.ones((3, 3))
    assert(calculate_agreement(population, 1, 1, 1) == 3), "Test 7"
    assert(calculate_agreement(population, 1, 1, -1) == 5), "Test 8"
    assert(calculate_agreement(population, 1, 1, 10) == 14), "Test 9"
    assert(calculate_agreement(population, 1, 1, -10) == -6), "Test 10"

    print("Tests passed")



def ising_main():
    """
    Main function to handle command-line arguments and launch the simulation or testing.
    """
    parser = argparse.ArgumentParser(description="Run the Ising model simulation.")
    parser.add_argument("-ising_model", action="store_true", help="Run the Ising model with default parameters")
    parser.add_argument("-external", type=float, default=0.0, help="External influence factor")
    parser.add_argument("-alpha", type=float, default=1.0, help="Alpha value for societal tolerance")
    parser.add_argument("-test_ising", action="store_true", help="Run the test functions for the model")
    args = parser.parse_args()

    if args.test_ising:
        test_ising()  # Run tests for the Ising model calculations
    elif args.ising_model:
        population = np.random.choice([-1, 1], size=(50, 50))
        model = {
            'population': population,
            'beta': 1 / args.alpha,
            'external': args.external
        }
        simulate_ising(model)  # Start the Ising model simulation




'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
==============================================================================================================
'''

def defuant_main():
	#Your code for task 2 goes here

def test_defuant():
	#Your code for task 2 goes here


'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''

def main():
	#You should write some code for handling flags here

if __name__=="__main__":
	main()