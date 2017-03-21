from util import *
from read_data import *
from sbm import *
import networkx as nx
import numpy as np
import pdb

# Load political blog dataset
n = 1000
L, E = gen_sym_sbm(n, 2, 0.4*n, 
				   0.00004*n, regime=SBM_LIN)
G = nx.Graph()
G.add_nodes_from(range(n))
G.add_edges_from(E)
for i in range(n):
	G.node[i]['value'] = L[i]
max_cc = max(nx.connected_components(G), key=len)
G = G.subgraph(max_cc)

# Compute relative cluster sizes
left, right = subgraphs_by_value(G, [0, 1])
num_left = 1.0*left.number_of_nodes()
num_right = 1.0*right.number_of_nodes()
num_total = num_left + num_right
p = np.array([num_left/num_total, num_right/num_total])

# Compute connectivity parameters
q_diag = np.array([get_conn_prob(left), get_conn_prob(right)])
remaining_edges = G.number_of_edges() - left.number_of_edges() - right.number_of_edges()
inter_prob = remaining_edges*1.0/(num_left*num_right)
Q = np.diag(q_diag)
Q[0,1] = Q[1,0] = inter_prob

print("Q:", Q)
print("p:", p)
n = G.number_of_nodes()
# Compute eigenvalues/vectors
PQ = np.matmul(np.diag(p),n*Q)
evals, evecs = np.linalg.eig(PQ)
print("Evecs, evals", evecs, evals)
# Parameters
c = 1e-3
k = 2
m = int(np.log(4*k)/min(p))
eps = 0.001
x = 1e-3
num_iters = int(np.log(n))

# run experiment
reliable_graph_classification(G, c, m, eps, x, num_iters, evals, evecs, p)
