from util import *
from read_data import *
import networkx as nx
import numpy as np
import pdb

# Load political blog dataset
G = graph_from_file("../data/polblogs.gml")

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

# Compute eigenvalues/vectors
evecs, evals = np.linalg.inv(np.diag(p)*Q)

# Parameters
c = 1e-3
k = 2
m = int(np.log(4*k)/min(p))
print(m)
eps = 0.9
n = G.number_of_nodes()
x = 1e-3
num_iters = int(np.log(n))

# run experiment
reliable_graph_classification(G, c, m, eps, x, num_iters, evals, evecs, p)
