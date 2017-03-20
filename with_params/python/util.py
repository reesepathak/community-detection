import networkx as nx
import math, random
from itertools import product as cprod

def basis(size, index):
	vec = np.zoers(size)
	vec[index] = 1.0
	return vec

def N_intersect(N_r, N_s, E):
	set_list = [N_r, N_s]
	return len(E.intersection(cprod(*setlist)))

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
	n = G.number_of_nodes()
	return 2*G.number_of_edges()/(1.0*n*(n-1))
	# degs = G.degree().values()
	# avg_num_neighbors = sum(degs)/(1.0 * len(degs))
	# return avg_num_neighbors/G.number_of_nodes()

def get_residual_network(G, vals):
	"""
	Remove in cluster edges
	"""
	res = G.copy()
	for val in vals:
		nodes = [i for (i,j) in res.nodes_iter(data=True) if j['value']==val]
		edges_to_remove = list(set(res.edges(nodes)) - set(res.edges()))
		for e in edges_to_remove:
			res.remove_edge(*e)
	return res

def intercluster_connectivity(res):
	"""
	Returns intercluster connectivity params
	Note: ONLY VALID in 2-cluster case
	"""
	num_edges = 1.0*res.number_of_edges()
	nodes1 = len([i for (i,j) in res.nodes_iter(data=True) if j['value']==1])
	nodes0 = len([i for (i,j) in res.nodes_iter(data=True) if j['value']==0])
	p = num_edges/(nodes1*nodes0)
	return [p, p]

def shortest_paths(r, G, v):
	paths = nx.single_source_shortest_path_length(G, v, cutoff=r)
	rnn = set([n for n in paths if paths[n] == r])
	return rnn

def vertex_comparison(u, v, r, s, edge_subset, x, c, evals, n, p, list_r, N_s):
    assert(0 <= c <= 1)
    eta = np.shape(evals)[1] if evals[-1] != 0 else np.shape(evals)[1]-1
    temp = np.transpose(np.vander((1-c)*evals, r + s + eta, increasing=True))
    
    # compute constraint matrix 
    matrix = temp[(r + s + 1):, (r + s + 1):]
    coeff = (((1-c)*n)/c)
    b_uv = coeff * [compute_N(r+j, s, edge_subset, u, v) for j in range(eta)]
    b_uu = coeff * [compute_N(r+j, s, edge_subset, u, u) for j in range(eta)]
    b_vv = coeff * [compute_N(r+j, s, edge_subset, v, v) for j in range(eta)]
    
    # LLS to compute z
    z_uv = np.linalg.solve(matrix, b_uv)
    z_uu = np.linalg.solve(matrix, b_uu)
    z_vv = np.linalg.solve(matrix, b_vv)

    # Step 2 of comparison
    p_min = np.min(np.array(p))
    lhs = z_uu - 2*z_uv + z_vv
    threshold = 5*(2*x*(1.0/np.sqrt(p_min)) + math.pow(x, 2))
    return len([i for i in range(eta) if lhs[i] > threshold]) > 0

def unreliable_graph_classification(G, c, m, eps, x, evals, evecs):
	
	def collect_shortest_path(threshold, m, resnet, verts):
		result = np.zeros((threshold, m))
		for t in range(1, threshold):
			for i in range(m):
				result[t, i] = shortest_paths(t, resnet, verts[i])
		return result

	eta = np.shape(evals)[0] if evals[-1] != 0 else (np.shape(evals)[0] - 1)
	# TODO: deal with underdetermined case
	assert(eta == np.shape(evals)[0])

	# grab random edge and vertex subset 
	edgeset = G.edges()
	n = G.number_of_nodes()
	sub_edges = set([])
	for e in edgeset:
		if random.random() <= c:
			subset.add(e)
	sub_verts = random.sample(G.nodes(), m)

	# compute parameters 
	val = np.log(n)/np.log(1.0*(1 - c)*evals[0])
	r = (1.0 - eps/3.0)*val - eta
	s = 2.0*(eps/3.0)*val
	
	# create residual network (remove selected edges)
	resnet = G.copy()
	resnet.remove_edges_from(sub_edges)
	N_table = collect_shortest_path(r + s, m, resnet, sub_verts)
	result_dict = {}
	for i in range(len(sub_verts)):
		for j in range(len(sub_vrts)):
			list_r = N_table[:, i]
			N_s = N_table[s, j]
			result_dict[i,j] = vertex_comparison(sub_verts[i], sub_verts[j], 
									   		     r, s, sub_edges, x, c, 
									   		     evals, n, p, 
									   		     list_r, N_s)

