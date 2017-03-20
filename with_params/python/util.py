import networkx as nx
import math, random
from itertools import product as cprod
import numpy as np

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

def vertex_comparison(u, v, r, s, E, x, c, evals, n, p, list_r, N_s):
    assert(0 <= c <= 1)
    eta = np.shape(evals)[1] if evals[-1] != 0 else np.shape(evals)[1]-1
    
    # compute constraint matrix 
    temp = np.transpose(np.vander((1-c)*evals, r + s + eta, increasing=True))
    matrix = temp[(r + s + 1):, (r + s + 1):]

    # compile vector constraints
    coeff = (((1-c)*n)/c)
    b_uv = coeff * [N_intersect(list_r[r+j,0], N_s[1], E) for j in range(eta)]
    b_uu = coeff * [N_intersect(list_r[r+j,0], N_s[0], E) for j in range(eta)]
    b_vv = coeff * [N_intersect(list_r[r+j,1], N_s[1], E) for j in range(eta)]
    
    # LLS to compute z
    z_uv = np.linalg.solve(matrix, b_uv)
    z_uu = np.linalg.solve(matrix, b_uu)
    z_vv = np.linalg.solve(matrix, b_vv)

    # Step 2 of comparison
    p_min = np.min(np.array(p))
    lhs = z_uu - 2*z_uv + z_vv
    threshold = 5*(2*x*(1.0/np.sqrt(p_min)) + math.pow(x, 2))
    return len([i for i in range(eta) if lhs[i] > threshold]) > 0

def vertex_classification(centers, v, N_list, r, s, E, evals, c, x, p, list_r, N_sv):
	coeff = 1.0*(1-c)*n
	coeff /= c
	z_cs = []
	z_vs = []
	for (i, vc) in enumerate(centers):
		temp = np.transpose(np.vander((1-c)*evals, r + s + eta, increasing=True))
		matrix = temp[(r + s + 1):, (r + s + 1):]
		b_v = coeff*[N_intersect(list_r[r+j, i], N_sv, E) for j in range(eta)]
		b_c = coeff*[N_intersect(list_r[r+j, i], list_r[s, i], E) for j in range(eta)]
		z_vs.append(np.linalg.solve(matrix, b_v))
		z_cs.append(np.linalg.solve(matrix, b_c))
	p_min = np.min(np.array(p))
	threshold = (19.0/3.0)*(2*x*(1.0/np.sqrt(p_min)) + math.pow(x, 2))
	for sig in range(len(centers)):
		for sig2 in range(len(centers)):
			if sig == sig2: 
				continue
			value = z_cs[sig] - z_cs[sig2] - 2*z_vs[sig] +2*z_vs[sig2]
			ret_val = True
			for elem in range(np.shape(value)[0]):
				if elem > threshold:
					ret_val = False 
					break
			if ret_val:
				return True
	return False


def unreliable_graph_classification(G, c, m, eps, x, evals, evecs, p):
	
	def collect_shortest_path(threshold, m, resnet, verts):
		print(threshold)
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
			sub_edges.add(e)
	sub_verts = random.sample(G.nodes(), m)

	# compute parameters 
	val = np.log(n)/np.log(1.0*(1 - c)*evals[0])
	r = (1.0 - (eps/3.0))*val - eta
	s = 2.0*(eps/3.0)*val
	print(r, s)
	
	# create residual network (remove selected edges)
	resnet = G.copy()
	resnet.remove_edges_from(sub_edges)
	N_table = collect_shortest_path(r + s, m, resnet, sub_verts)
	result_dict = {}
	for i in range(m):
		for j in range(m):
			list_r = N_table[:, (i,j)]
			N_s = [N_table[s, i], N_table[s, j]]
			result_dict[i,j] = vertex_comparison(sub_verts[i], 
												 sub_verts[j], 
									   		     r, s, sub_edges, x, c, 
									   		     evals, n, p, 
									   		     list_r, N_s)
	# assume k = 2 
	assert(eta == 2)
	
	# Step 6
	centers = None
	for i in range(m):
		for j in range(m):
			if not result_dict[i,j]:
				centers = [sub_verts[i], sub_verts[j]]
	assert(centers)

	# Step 7
	full_table = collect_shortest_path(s, n, resnet, G.nodes())
	final_classes = []
	for v in G.nodes():
		list_r = full_table[:, centers]
		N_sv = full_table[s, v]
		final_classes.append(vertex_classification(centers, v, 
												 r, s, sub_edges,
												 evals, c, x, p, list_r, N_sv))
	return final_classes

def reliable_graph_classification(G, c, m, eps, x, num_iters, evals, evecs, p):
	unreliables = []
	
	def disagreement(assign1, assign2):
		size = np.shape(assign1)[0]
		return (n - np.max(np.shape(np.where(assign1 == assign2))[0],
			  		   		  np.shape(np.where(assign1 != assign2))[0]))/(n*1.0)

	for _ in range(num_iters):
		unreliables.append(unreliable_graph_classification(G, c, m, eps, x, evals, evecs, p))

	good_assigns = []
	threshold = 4*len(evals)*np.exp(-1.0*(1-c)*(evals[-1]**2)*min(p)/(16*evals[0]*2*(1 + x)))
	threshold /= (1 - exp((-1.0*(1-c)*(evals[-1]**2)*min(p)/(16*evals[0]*2*(1 + x)))*((1-c)*math.pow(evals[-1], 4)/(4*math.pow(evals[0],3)))))
	for res in unreliables:
		num_disagreement = 0
		for res2 in unreliables:
			if res == res2:
				continue
			dis = disagreement(res, res2)
			if dis > threshold:
				num_disagreement += 1
		if num_disagreement <= 0.5*(len(unreliables) - 1):
			good_assigns.append(res)
	good_assigns = np.array(good_assigns)
	final_assign = []
	for i in range(np.shape(good_assigns)[1]):
		final_assign.append(random.choice(list(good_assigns[:, i])))
	return final_assign

