import random 
import numpy as np
from math import log, ceil, sqrt
from itertools import combinations
import networkx as nx
import sbm
import matplotlib.pyplot as plt

random.seed(1)

with open('data/polblogs.gml', 'r') as f:
    data = f.readlines()

raw_edges = set()
labels = {}
for row, line in enumerate(data):
    if ' node ' in line:
        v = int(data[row+1].strip().split()[1])
        labels[v] = int(data[row+3].strip().split()[1])

    if ' edge ' in line:
        v1 = int(data[row+1].strip().split()[1])
        v2 = int(data[row+2].strip().split()[1])
        if v1 == v2: continue
        e = (v1, v2) if v1 < v2 else (v2, v1)
        raw_edges.add(e)

# use networkx to find the largest component
G = nx.Graph()
G.add_edges_from(raw_edges)
Gc = max(nx.connected_component_subgraphs(G), key=len)
edges = set(Gc.edges())
nodes = Gc.nodes()
N = len(nodes)
C1 = len({v for v in nodes if labels[v] == 1})
C0 = len({v for v in nodes if labels[v] == 0})

total_in1 = 0
total_in0 = 0
total_out = 0
for (v1, v2) in edges:
    if labels[v1] == labels[v2] == 1: 
        total_in1 += 1
    elif labels[v1] == labels[v2] == 0:
        total_in0 += 1
    else:
        total_out += 1

def choose_2(k): return k * (k - 1) / 2.0

p_in1 = float(total_in1) / choose_2(C1)
p_in0 = float(total_in0) / choose_2(C0)
p_out = float(total_out) / (choose_2(N) - choose_2(C1) - choose_2(C0))

# p is probability of a node being in each community
p_0 = C0 / float(N)
p_1 = C1 / float(N)
P = np.matrix(np.diag([p_0, p_1]))
Q = np.matrix(np.zeros((2, 2)))
Q[0, 0] = p_in0 
Q[1, 1] = p_in1
Q[0, 1] = p_out
Q[1, 0] = p_out
Q *= (N / log(N))

PQ = P * Q
theta_0 = np.array(PQ[:, 0])
theta_1 = np.array(PQ[:, 1])

ts = np.arange(0.0, 1.01, 0.001)
CHs = np.zeros(len(ts))
for i, t in enumerate(ts):
    CHs[i] = np.sum(t * theta_0 + (1 - t) * theta_1 - theta_0 ** t * theta_1**(1-t))
best = np.argmax(CHs)
print "Maximized at t=%f: %f" % (ts[best], CHs[best])

