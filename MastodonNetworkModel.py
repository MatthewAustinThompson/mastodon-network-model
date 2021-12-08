#!/usr/bin/env python3

# # Creating Network Model of Mastodon 

# **Questions:**
# - Can user be a part of instance without any edges? Done via node label. If not, makes assumption that all users who join an instance immediately follow someone within that instance. Making this assumption allows for more clear visualizations because a node "in" an instance whose first edge is outside that instance will be positioned within a that outside instance rather than it's own

# **Goal Network Dynamics:** <br>
# Scale free (Barabos?), aka power law, within an instance, small world-like probability of rewiring an edge to ANY node w/ scale free probability

# **Overview of Possible Steps:** <br>
# Start with **set number of instances (currently trying this approach)** or just 1 node and add instances during simulation? For each time step: 
# - Add a node
# - Determine which "instance" node should be in using rich-get-richer dynamics
# - Pick One: **Create edge to node within that instance  (currently trying this approach)**; or just label node as part of that instance
# - Select a random node
# - Use small-world probability to determine whether that node creates edge to another node within instance or outside of instance
# - Randomly determine edge target, considering either population of nodes within instance or population of nodes outside instance

# ## Import Packages

import networkx as nx
import numpy as np
#import plotly.graph_objects as go
import matplotlib.pyplot as plt
import multiprocessing, multiprocessing.pool
from scipy.stats import expon, powerlaw
from itertools import count
from collections import defaultdict
from tqdm import tqdm # Progress bar

NUM_WORKERS = 10       # Number of processes
WORKER_CHUNKSIZE = 50  # Number of configurations to hand out at once

# Takes an array of integers for nodes count in each instance
# returns (networkx network, {instance: set(nodes in instance)})
# This creates a network with appropriate number of nodes and instance
# attributes, but does not create any edges
def create_empty_network(nodes_per_instance):
	num_instances = len(nodes_per_instance)
	instances = {}
	G = nx.Graph()

	nodes_created = 0
	for i in range(0, num_instances):
		instances[i] = set()

		# Adds all nodes for each instance
		for n in range(0, nodes_per_instance[i]):
			G.add_node(nodes_created)
			G.nodes[nodes_created]['state'] = 0
			G.nodes[nodes_created]['instance'] = i
			instances[i].add(nodes_created)
			nodes_created += 1
	return (G, instances)

# Takes an empty graph and a set of nodes in an instance
# creates edges between those nodes according to preferential attachment
def create_edges_in_instance(G, instance, power_law_exponent=2.5):
	nodes = list(instance)
	# Find appropriate degree distribution, simulate a graph of just this instance
	degree_distribution = nx.utils.powerlaw_sequence(len(nodes), power_law_exponent)
	expected_graph = nx.expected_degree_graph(degree_distribution, selfloops=False)
	# Now mimick the simulated graph, adding the corresponding edges to our own graph
	for i,node in enumerate(nodes):
		peers = list(expected_graph.neighbors(i))
		for peer in peers:
			destination = nodes[peer]
			G.add_edge(node, destination)

# Takes a graph, and a value from (0..1), returns
# the closest node in the degree sequence by percentile.
# For example, if given ".26", finds the closest to the 26%th node
# by degree distribution. 0 returns the lowest degree node, 1 the highest
def get_node_by_degree_percentile(G, percentile):
	distribution = sorted(list(G.degree()), key=lambda d: d[1])
	index = round(percentile * len(distribution))
	if( index < 0 ):
		index = 0
	if( index == len(distribution) ):
		index -= 1
	return distribution[index][0]

def rewire_edges(G, rewire_probability, power_law_exponent=2.5):
	edges = list(G.edges())
	rng = np.random.default_rng()
	sequence = list(rng.power(power_law_exponent, len(edges)))
	for edge in edges:
		if( rng.random() < rewire_probability ):
			# Find new destination
			new_dest = get_node_by_degree_percentile(G, sequence.pop())
			
			# If the new destination is the same as the old, or a self-loop,
			# do nothing
			(src,dst) = edge
			if( new_dest == dst or new_dest == src ):
				continue

			# Otherwise, replace the current edge with the new edge
			G.remove_edge(src,dst)
			G.add_edge(src, new_dest)

def remove_disconnected_nodes(G):
	to_remove = [node for node,degree in dict(G.degree()).items() if degree == 0]
	G.remove_nodes_from(to_remove)

# On average, how many instances is each node connected to?
# Returns as a percentage of total instances
def get_avg_instances_connected_to_nodes(G, total_instances):
	connectivity = []
	for node in G.nodes().keys():
		instances = set()
		for neighbor in G.neighbors(node):
			instances.add(G.nodes[neighbor]["instance"])
		connectivity.append(len(instances)/total_instances)
	return np.mean(connectivity)

# See: http://www.countrysideinfo.co.uk/simpsons.htm
def get_simpson_follower_diversity_index_single_node(G, node):
	neighboring_instances = defaultdict(lambda: 0)
	for neighbor in G.neighbors(node):
		instance = G.nodes[neighbor]["instance"]
		neighboring_instances[instance] += 1
	total_neighbors = sum(neighboring_instances.values())
	diversity = 0.0
	for instance,count in neighboring_instances.items():
		diversity += ((count / total_neighbors)**2)
	return 1-diversity

# What's the probability that two followers are on the same instance?
def get_simpson_follower_diversity_index(G):
	diversity_scores = []
	for node in G.nodes().keys():
		diversity_scores.append(get_simpson_follower_diversity_index_single_node(G, node))
	return np.mean(diversity_scores)

# What percentage of nodes are reachable from each node, on average?
def get_average_reachability(G):
	reachable = []
	total_nodes = len(G.nodes())
	for node in G.nodes().keys():
		from_here = len(nx.descendants(G, node))
		reachable.append(from_here/total_nodes)
	return np.mean(reachable)

def render_graph(G, title, filename):
	# Map colors
	groups = set(nx.get_node_attributes(G, 'instance').values())    # Groups colors by node's instance
	# groups = set(nx.get_node_attributes(G, 'state').values())    # Groups colors by node's degree
	mapping = dict(zip(sorted(groups), count()))
	colors = [mapping[G.nodes[n]['instance']] for n in G.nodes()]

	# Display network
	nx.draw(G, with_labels=False, node_size=200, pos=nx.fruchterman_reingold_layout(G), node_color=colors)
	plt.title(title)
	plt.savefig(filename, bbox_inches="tight")
	plt.clf()

def run_simulation(configuration):
	(size_distribution_name, nodes_per_instance, num_instances, p, power_law, trial) = configuration
	(G, instances) = create_empty_network(nodes_per_instance)
	for instance in instances.values():
		create_edges_in_instance(G, instance, power_law_exponent=power_law)
	rewire_edges(G, p, power_law)
	remove_disconnected_nodes(G)
	instance_connectivity = get_avg_instances_connected_to_nodes(G, num_instances)
	simpsons_index = get_simpson_follower_diversity_index(G)
	reachability = get_average_reachability(G)
	if( trial == 0 ):
		title = "%d Instances, instance sizes=%s, p=%.2f, exponent=%.2f" % (num_instances, size_distribution_name, p, power_law)
		filename = "test_%s_p_%.2f_e_%.1f_.png" % (size_distribution_name, p,power_law)
		render_graph(G, title, filename)
	return (size_distribution_name, p, power_law, instance_connectivity, simpsons_index, reachability)

if __name__ == "__main__":
	num_instances = 10
	p_step = 100
	pl_step = 5
	trials = 5

	# Define instance sizes
	equal_instances = np.full(num_instances, 40)
	expon_instances = (np.linspace(expon.ppf(0.1), expon.ppf(0.5), num_instances) * 100).astype(int)
	gamma = 0.67
	pwrlaw_instances = ((np.linspace(100, 1, num_instances) ** -gamma) * 100).astype(int)
	nodes_per_instance = {"equal":equal_instances, "expon":expon_instances, "pwrlaw":pwrlaw_instances}
	instance_keys = list(nodes_per_instance.keys())

	log = open("simulation_log.csv", "w")
	log.write("nodes_per_instance,p,exponent,instance_connectivity,simpsons_index,reachability\n")
	pool = multiprocessing.get_context("spawn").Pool(processes=NUM_WORKERS)
	configurations = []
	for size_name in instance_keys:
		for p in np.geomspace(0.0001,1,p_step):
			for power_law in np.linspace(1.5, 3, pl_step):
				for trial in range(0, trials):
					configuration = [size_name, nodes_per_instance[size_name], num_instances, p, power_law, trial]
					configurations.append(configuration)
	results = list(tqdm(pool.imap(run_simulation, configurations, chunksize=WORKER_CHUNKSIZE), total=len(configurations), desc="Running simulations"))
	for result in results:
		log.write("%s,%.5f,%.2f,%.5f,%.5f,%.5f\n" % result)
	log.close()
