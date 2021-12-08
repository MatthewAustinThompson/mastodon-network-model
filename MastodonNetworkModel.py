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
from scipy.stats import expon, powerlaw
from itertools import count
from collections import defaultdict
from tqdm import tqdm # Progress bar

# ## Set Plotly Options

def do_plotly(G, pos=None):
	# Default Network Layout
	if pos == None:
		pos=nx.fruchterman_reingold_layout(G)

		
	# Create Edges
	edge_x = []
	edge_y = []
	for edge in G.edges():
		# x0, y0 = G.nodes[edge[0]]['pos']
		x0, y0 = pos[edge[0]]
		x1, y1 = pos[edge[1]]
		edge_x.append(x0)
		edge_x.append(x1)
		edge_x.append(None)
		edge_y.append(y0)
		edge_y.append(y1)
		edge_y.append(None)

	edge_trace = go.Scatter(
		x=edge_x, y=edge_y,
		line=dict(width=0.5, color='#888'),
		hoverinfo='none',
		mode='lines')

	node_x = []
	node_y = []
	for node in G.nodes():
		# x, y = G.nodes[node]['pos']
		x, y = pos[node]
		node_x.append(x)
		node_y.append(y)

	node_trace = go.Scatter(
		x=node_x, y=node_y,
		mode='markers',
		hoverinfo='text',
		marker=dict(
			showscale=True,
			# Colorscale options
			# 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
			# 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
			# 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
			colorscale='YlGnBu',
			reversescale=True,
			color=[],
			size=10,
			colorbar=dict(
				thickness=15,
				title='Node Connections',
				xanchor='left',
				titleside='right'
			),
			line_width=2))

	
	# Color Node Point
	node_adjacencies = []
	node_text = []
	for node, adjacencies in enumerate(G.adjacency()):
		node_adjacencies.append(len(adjacencies[1]))
		node_text.append('label: '+str(node)+'# of connections: '+str(len(adjacencies[1])))

	node_trace.marker.color = node_adjacencies
	node_trace.text = node_text

	
	# Create Network Graph
	fig = go.Figure(data=[edge_trace, node_trace],
				layout=go.Layout(
					title='<br>Network graph made with Python',
					titlefont_size=16,
					showlegend=False,
					hovermode='closest',
					margin=dict(b=20,l=5,r=5,t=40),
					annotations=[ dict(
						text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
						showarrow=False,
						xref="paper", yref="paper",
						x=0.005, y=-0.002 ) ],
					xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
					yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
				)
	#fig.show()
	fig.write_image("test.png")

# ## Define Function to Select Instance to Create Node Within 

def preferential_attachment(G, stubs, instances, p): 
	# ADDING NODE
	# Add new node
	new_node = G.number_of_nodes()
	G.add_node(new_node)
	
	# Determine which instance to add to
	# Randomly choose from list of instances
#     chosen_instance = np.random.choice(list(instances.keys()))
	# OR
	# Randomly select node and choose the instance that node is in 
	chosen_instance = G.nodes[np.random.choice(stubs)]['instance']
	
	# Select random node within instance to create edge to
	neighbor = np.random.choice(instances[chosen_instance])
	
	# Add edge
	G.add_edge(new_node, neighbor)
	
	# Update attributes and history 
	G.nodes[new_node]['state'] = 1
	G.nodes[neighbor]['state'] += 1
	G.nodes[new_node]['instance'] = chosen_instance
	instances[chosen_instance].append(new_node)
	stubs.append(new_node)
	stubs.append(neighbor)
	
	
	# REWIRING
	# Select random node to make edge from
	rand_node = np.random.choice(stubs)
	rand_node_instance = G.nodes[rand_node]['instance']
	
	# If probability met, attach to random node in external instance, else, 
	# attach inside current instance
	possible_nodes = []
	rand_num = np.random.uniform(0, 1)
	if rand_num <= p:
		external_instances = list(instances.keys())
		external_instances.remove(rand_node_instance)
		for i in range(0, len(external_instances)):
			for j in range(0, len(instances[external_instances[i]])):
				possible_nodes.append(instances[external_instances[i]][j])
		neighbor = np.random.choice(possible_nodes)
	else:
		neighbor = np.random.choice(instances[rand_node_instance])
		
	# Add edge
	G.add_edge(rand_node, neighbor)
	
	# Update attributes and history 
	G.nodes[rand_node]['state'] += 1
	G.nodes[neighbor]['state'] += 1
	stubs.append(rand_node)
	stubs.append(neighbor)
	
	
	return(G, stubs, instances)

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

	bar = tqdm(desc="Running simulations", total=p_step*pl_step*trials*len(instance_keys))
	log = open("simulation_log.csv", "w")
	log.write("nodes_per_instance,p,exponent,instance_connectivity,simpsons_index,reachability\n")
	for sizes in range(0, len(instance_keys)):
		for p in np.geomspace(0.0001,1,p_step):
			for power_law in np.linspace(1.5, 3, pl_step):
				for trial in range(0, trials):
					(G, instances) = create_empty_network(nodes_per_instance[instance_keys[sizes]])
					for instance in instances.values():
						create_edges_in_instance(G, instance, power_law_exponent=power_law)
					rewire_edges(G, p, power_law)
					remove_disconnected_nodes(G)
					instance_connectivity = get_avg_instances_connected_to_nodes(G, num_instances)
					simpsons_index = get_simpson_follower_diversity_index(G)
					reachability = get_average_reachability(G)
					log.write("%s,%.2f,%.2f,%.5f,%.5f,%.5f\n" % (instance_keys[sizes], p, power_law, instance_connectivity, simpsons_index, reachability))
					if( trial == 0 ):
						title = "%d Instances, instance sizes=%s, p=%.2f, exponent=%.2f" % (num_instances, instance_keys[sizes], p, power_law)
						filename = "test_%s_p_%.2f_e_%.1f_.png" % (instance_keys[sizes], p,power_law)
						render_graph(G, title, filename)
					bar.update(1)
	bar.close()
	log.close()
