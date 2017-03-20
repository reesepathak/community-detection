import networkx as nx
import math

def basis(size, index):
	vec = np.zoers(size)
	vec[index] = 1.0
	return vec

def compute_N(r, s, E, sigu, sigv, c, evals, evecs, Pinv, n):
	coeff = (1.0/n)*c*math.pow(1-c, r+s)
	result = 0.0
	dim = np.shape(evals)[0]
	basis_u = basis(dim, sigu)
	basis_v = basis(dim, sigv)
	for i in range(dim):
		proj_u = np.dot(basis_u, evec[:, i])*evec[:, i]
		proj_v = np.dot(basis_v, evec[:, i])*evec[:, i]
		result += math.pow(evals[i], r + s + 1)*np.dot(proj_u, Pinv*proj_v)
	return coeff*result

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
