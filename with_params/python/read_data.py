import networkx as nx 

def graph_from_file(fname):
	"""
	Reads in a graph from file
	Args:
	    fname: path to file
	Returns: LightGraph object
	"""
	orig_graph = nx.read_gml(fname)
	# obtain list of weakly connected subgraphs
	list_of_WCCs = nx.weakly_connected_component_subgraphs(orig_graph)
	# obtain index of largest such weakly connected component
	G = max(list_of_WCCs, key=len)
	return nx.Graph(G)