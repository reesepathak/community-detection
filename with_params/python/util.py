import networkx as nx

def subgraphs_by_value(G, vals):
	"""
	Get subgraphs by node values
	"""
	subgraphs = []
	for val in vals:
		nodes = [i for (i,j) in G.nodes_iter(data=True) if j['value']==val]
		subgraphs.append(G.subgraph(nodes))
	return subgraphs

def get_conn_prob(G):
	degs = G.degree().values()
	avg_num_neighbors = sum(degs)/(1.0 * len(degs))
	return avg_num_neighbors/G.number_of_nodes()
