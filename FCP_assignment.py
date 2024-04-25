import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

	while False:
			def get_mean_degree(self):
				#Your code  for task 3 goes here

				def get_mean_clustering(self):
				#Your code for task 3 goes here

					def get_mean_path_length(self):
				#Your code for task 3 goes here

						def make_random_network(self, N, connection_probability):
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

	def make_ring_network(self, N, neighbour_range=2):
		"""
		Create a ring network of nodes according to a given range of neighbours. 
		
		Arg: 
			N (int): The number of nodes in the network.
			neighbour_range (int): The range of neighbors each node connects to (default is 2).
		"""
		
		# Create an empty list for nodes
		self.nodes = []

		# Loop through each node
		for i in range(N):
			connections = []

			# loop for setting connections of each node
			for j in range(1, neighbour_range + 1):
				prev_index = i - j
				next_index = i + j
				connections.append(next_index)
				connections.append(prev_index)

			# Add the node to the network with its connections
			self.nodes.append(Node(value=i, number=i, connections=connections))  # Set connections attribute of node

	def make_small_world_network(self, N, re_wire_prob=0.2):
		"""
		Create a small-world network from a ring network by re-wiring edges according to a set probability. 
		
		Args: 
			N (int): The number of nodes in the network.
			re_wire_prob (float): The rewiring probability (default is 0.2).
		"""
		
		# Start with a ring network
		self.make_ring_network(N)

		# Loop through each node
		for node in self.nodes:

			for i, neighbour_index in enumerate(node.connections):

				# Using numpy's random generator for handling probabilities
				if np.random.rand() < re_wire_prob:

					# Get indices of current neighbours
					current_neighbour_indices = set(node.connections)

					# Choose a new neighbour from all nodes except itself and its current neighbours
					new_neighbour_index = np.random.choice([x for x in range(N) if x != node.index and x not in current_neighbour_indices])  # Choose a new neighbour from all nodes except itself and its current neighbours
					node.connections[i] = new_neighbour_index

	def plot(self, network_type=None, re_wire_prob=None):
		"""
        Plot the network.

        Args:
            network_type (str): Type of network (e.g., "Ring", "Small-World").
            re_wire_prob (float): Rewiring probability for small-world networks.
        """

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
					
while False:
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
		'''
		This function should return the *change* in agreement that would result if the cell at (row, col) was to flip it's value
		Inputs: population (numpy array)
				row (int)
				col (int)
				external (float)
		Returns:
				change_in_agreement (float)
		'''

		#Your code for task 1 goes here

		return np.random * population

	def ising_step(population, external=0.0):
		'''
		This function will perform a single update of the Ising model
		Inputs: population (numpy array)
				external (float) - optional - the magnitude of any external "pull" on opinion
		'''
		
		n_rows, n_cols = population.shape
		row = np.random.randint(0, n_rows)
		col  = np.random.randint(0, n_cols)

		agreement = calculate_agreement(population, row, col, external=0.0)

		if agreement < 0:
			population[row, col] *= -1

		#Your code for task 1 goes here

	def plot_ising(im, population):
		'''
		This function will display a plot of the Ising model
		'''

		new_im = np.array([[255 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)
		im.set_data(new_im)
		plt.pause(0.1)

	def test_ising():
		'''
		This function will test the calculate_agreement function in the Ising model
		'''

		print("Testing ising model calculations")
		population = -np.ones((3, 3))
		assert(calculate_agreement(population,1,1)==4), "Test 1"

		population[1, 1] = 1.
		assert(calculate_agreement(population,1,1)==-4), "Test 2"

		population[0, 1] = 1.
		assert(calculate_agreement(population,1,1)==-2), "Test 3"

		population[1, 0] = 1.
		assert(calculate_agreement(population,1,1)==0), "Test 4"

		population[2, 1] = 1.
		assert(calculate_agreement(population,1,1)==2), "Test 5"

		population[1, 2] = 1.
		assert(calculate_agreement(population,1,1)==4), "Test 6"

		"Testing external pull"
		population = -np.ones((3, 3))
		assert(calculate_agreement(population,1,1,1)==3), "Test 7"
		assert(calculate_agreement(population,1,1,-1)==5), "Test 8"
		assert(calculate_agreement(population,1,1,-10)==14), "Test 9"
		assert(calculate_agreement(population,1,1,10)==-6), "Test 10"

		print("Tests passed")


	def ising_main(population, alpha=None, external=0.0):
		
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_axis_off()
		im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

		# Iterating an update 100 times
		for frame in range(100):
			# Iterating single steps 1000 times to form an update
			for step in range(1000):
				ising_step(population, external)
			print('Step:', frame, end='\r')
			plot_ising(im, population)


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
	"""
    Parse command line arguments and create/plot a ring network or a small-world network based on the provided arguments.
    """

	# Create argument parser
	parser = argparse.ArgumentParser(description='Create and plot a ring network of size N or a small-world network of size N -with a rewiring probability.')
	
	# Define command line arguments
	parser.add_argument('-ring_network', type=int, help='Specify the size of the ring network')
	parser.add_argument('-small_world', type=int, help='Specify the size of the small-world network')
	parser.add_argument('-re_wire', type=float, default=0.2, help='Specify the rewiring probability (default is 0.2)')

	# Parse the arguments
	args = parser.parse_args()

	# Create a network instance
	network = Network()

	# Check which type of network to create based on the provided arguments
	if args.ring_network:
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

	else:
		# Prompt user to specify the size of the network using appropriate flags
		print('Please specify the size of the network using the -ring_network or -small_world flag.')

if __name__ == '__main__':
    main()