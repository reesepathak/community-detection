import random 
import numpy as np
from math import log, ceil
from itertools import combinations
import networkx as nx

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
max_node = max(nodes)

d = 2.0 * len(edges) / len(nodes)

# currently mysterious how to set r'
# constraints from the NIPS paper: 
#   r = 3.0 / 4.0 * log(n) / log(d)
#   r + r' >= log n / log | (a - b) / k |
#   r + r' odd
# we suspect r' should be less than r because of its form in the general case
global_r = 2
global_r_ = 1 

c = 0.1

E = {e for e in edges if random.random() < c}
g_minus_e = edges - E

adj = {}
for v in nodes:
    adj[v] = set()
for (v1, v2) in g_minus_e:
    adj[v1].add(v2)
    adj[v2].add(v1)

high_degree_nodes = [n for n in nodes if len(adj[n]) > d]

def nbors(r, v):
    """aka N_{r[G\E]}(v)"""
    depth = np.zeros(max_node + 1, dtype=int)
    visited = np.zeros(max_node + 1, dtype=bool)
    bfs_queue = [v]
    visited[v] = True

    ok = []
    while len(bfs_queue) > 0:
        front = bfs_queue.pop()
        for nbor in adj[front]:
            if not visited[nbor]:
                visited[nbor] = True

                depth[nbor] = depth[front] + 1

                if depth[nbor] == r:
                    ok.append(nbor)
                else:
                    bfs_queue.append(nbor)
    return ok

def mutual_nbor_count(r, r_, v, v_):
    """aka N_{r, r' [E]}(v, v_) from the NIPS paper"""
    
    count = 0
    for v1 in nbors(r, v):
        for v2 in nbors(r_, v_):
            if (v1, v2) in E or (v2, v1) in E:
                count +=1
    return count

def mutual_nbor_frac1(r, r_, v, v_):
    """ a guess at the meaning of N' from arxiv.org/pdf/1506.03729.pdf
        where we just normalize N by number of pairs (v1, v2) considered
    """
    total = 0
    hits = 0
    for v1 in nbors(r, v):
        for v2 in nbors(r_, v_):
            total += 1
            if (v1, v2) in E or (v2, v1) in E:
                hits +=1
    return float(hits) / total if total > 0 else 0.0
    
bi_edges = edges | {(b, a) for (a, b) in edges}
def mutual_nbor_frac2(r, r2, v, v2):
    """ a guess at the meaning of N' from arxiv.org/pdf/1506.03729.pdf
        where we interpret the following description literally:

        the fraction of pairs of an edge leaving the ball of radius r centered on v 
        and an edge leaving the ball of radius r' centered on v'
        which hit the same vertex but are not the same edge

        This is slow and doesn't perform well.
    """
    v_ball = set(nbors(r, v))
    v_ball_nbors = set(nbors(r+1, v))

    v2_ball = set(nbors(r2, v2))
    v2_ball_nbors = set(nbors(r2+1, v2))

    # keep oriented: (node inside ball, node outside ball)
    leaving_v_ball = [(u, w) for (u, w) in bi_edges if u in v_ball and w in v_ball_nbors]
    leaving_v2_ball = [(u, w) for (u, w) in bi_edges if u in v2_ball and w in v2_ball_nbors]

    # count collisions on distinct edges
    total = 0
    hits = 0
    for (u1, w1) in leaving_v_ball:
        for (u2, w2) in leaving_v2_ball:
            total += 1
            if w1 == w2 and u1 != u2:
                hits += 1

    return float(hits) / total if total > 0 else 0.0

def sign_stat(v, v_):
    """aka I_{global_r, global_r, [E]}(v, v_)"""
    x = mutual_nbor_frac1(global_r, global_r_, v, v_)
    y = mutual_nbor_frac1(global_r+1, global_r_, v, v_)
    z = mutual_nbor_frac1(global_r+2, global_r_, v, v_)
    return x*z - y*y

def partition(vertices, stats):
    """ Returns a consistent partition of vertices into two groups:
        i.e. stats[i, j] > 0 iff i and j are in the same group
        Throws an exception if none exists.
    """
    assert len(vertices) == stats.shape[0] == stats.shape[1]
    vertices = np.array(vertices)
    signs = (stats > 0)
    indices = np.arange(len(vertices))

# try to find a partition such that sign_stat(v_i, v_j) > 0 
#   iff v_i and v_j are in the same community
#   if we can't, fail
#   if we can, pick a random node from each community to be our anchor
#

# select 5 vertices randomly and compute pairwise sign statistics
v = random.sample(xrange(n), 5)
sign_stats = np.zeros((len(v), len(v)))
for i in range(len(v)):
    for j in range(len(v)):
        sign_stats[i][j] = sign_stat(v[i], v[j])


print "sign_stats", sign_stats.astype(int)

# check if there exists a consistent partition of these vertices
#   i.e. sign_stat[v_i, v_j] > 0 iff v_i and v_j are in the same community
for i in range(2**len(v)):
    split1 = []
    split2 = []
    j = i
    ctr = 0
    while j:
        if j%2:
            split1.append(ctr)
        else:
            split2.append(ctr)
        ctr = ctr+1
        j >>= 1
    if not len(split1) or not len(split2):
        continue
    partition = True
    for u in split1:
        for v in split2:
            if sign_stats[u][v] > 0:
                partition = False
    for u in split1:
        for v in split1:
            if sign_stats[u][v] <= 0:
                partition = False
    for u in split2:
        for v in split2:
            if sign_stats[u][v] <= 0:
                partition = False
    if not np.all(signs == signs.T):
        raise ValueError("Signs not symmetric")

    A = [indices[0]]
    B = []
    
    for i in indices[1:]:
        if np.all(signs[i,A]) and not np.any(signs[i,B]):
            A.append(i)
        elif np.all(signs[i,B]) and not np.any(signs[i,A]):
            B.append(i)
        else:
            raise ValueError("No consistent way to assign %s into %s or %s" % 
                (vertices[i], vertices[A], vertices[B]))

    if len(B) == 0:
        raise ValueError("All of the nodes are in the same partition")

    return (vertices[A], vertices[B])

def anchor_by_partition():
    """ Pick 4 vertices, then try to find a consistent partition based on sign_stats.

        This is the algorithm from the NIPS paper.

        Currently we just try it a few times to check the success rate against the true labeling.
    """
    correct_partitions = 0
    incorrect_partitions = 0
    no_partitions = 0

    trial_count = 500
    for trial in range(trial_count):
        print "trial", trial
        vs = random.sample(high_degree_nodes, 4)
        sign_stats = np.zeros((len(vs), len(vs)))
        for i in range(len(vs)):
            for j in range(len(vs)):
                sign_stats[i][j] = sign_stat(vs[i], vs[j])

        try: 
            C1, C2 = partition(vs, sign_stats)
            C1_labels = np.array([labels[v] for v in C1])
            C2_labels = np.array([labels[v] for v in C2])

            if ((np.all(C1_labels == 0) and np.all(C2_labels == 1)) 
                or (np.all(C1_labels == 1) and np.all(C2_labels == 0))):
                correct_partitions += 1
            else:
                incorrect_partitions += 1
        except ValueError as e:
            no_partitions += 1
    print "Out of %d trials:" % trial_count
    print "\t %d returned a true partition" % correct_partitions
    print "\t %d returned a false partition" % incorrect_partitions
    print "\t %d didn't find a partition" % no_partitions

def anchor_by_avg():
    """ Follows arxiv.org/pdf/1506.03729.pdf by randomly selecting two high-degree nodes
    and checking if their N' is less than average.

    The paper doesn't say how to find the average, but we can estimate it by sampling other 
    high-degree nodes.

    Currently we just try it a bunch of times to check the success rate against the true labeling.
    """
    complete_correct_trials = 0
    complete_incorrect_trials = 0
    rejected_inconsistent_trials = 0
    rejected_same_side_correct_trials = 0
    rejected_same_side_incorrect_trials = 0

    trial_count = 200
    estimate_count = 100

    print "estimating avg mutual nbor frac among nodes"
    total_frac = 0.0
    for estimate in range(estimate_count):
        print "estimate", estimate
        v1, v2 = random.sample(high_degree_nodes, 2)
        total_frac += mutual_nbor_frac1(global_r, global_r_, v1, v2)
    avg_frac = total_frac / estimate_count

    for trial in range(trial_count):
        print "trial:", trial
        v1, v2 = random.sample(high_degree_nodes, 2)

        if (mutual_nbor_frac1(global_r, global_r_, v1, v2) < avg_frac and 
            mutual_nbor_frac1(global_r, global_r_, v2, v1) < avg_frac):
             # algo thinks different
            if labels[v1] != labels[v2]:
                complete_correct_trials += 1
            else:
                complete_incorrect_trials += 1
        else:
            # algo thinks same
            if labels[v1] == labels[v2]:
                rejected_same_side_correct_trials += 1
            else:
                rejected_same_side_incorrect_trials += 1
    print "Out of %d trials:" % trial_count
    print "\t %d returned a true partition" % complete_correct_trials
    print "\t %d returned a false partition" % complete_incorrect_trials
    print "\t %d rejected because it correctly thought the nodes were on the same side" % rejected_same_side_correct_trials
    print "\t %d rejected because it incorrectly thought the nodes were on the same side" % rejected_same_side_incorrect_trials

def anchor_by_sign_stats():
    """ Randomly select 2 high-degree nodes and check if they have opposite sign_stat.
    This is halfway between the NIPS paper and arxiv.org/pdf/1506.03729.pdf, and
    performs the best so far.

    Currently we just try it a bunch of times to check the success rate against the true labeling.
    """
    complete_correct_trials = 0
    complete_incorrect_trials = 0
    rejected_inconsistent_trials = 0
    rejected_same_side_correct_trials = 0
    rejected_same_side_incorrect_trials = 0

    trial_count = 200
    for trial in range(trial_count):
        print "trial", trial
        v1, v2 = random.sample(high_degree_nodes, 2)

        sign1 = (sign_stat(v1, v2) > 0)
        sign2 = (sign_stat(v2, v1) > 0)
        if (sign1 != sign2):
            rejected_inconsistent_trials += 1
            continue
        if (sign1 == sign2 == True):
            if labels[v1] == labels[v2]:
                rejected_same_side_correct_trials += 1
            else:
                rejected_same_side_incorrect_trials += 1
            continue

        # if we got to here, algo thinks they are different partitions 
        if labels[v1] != labels[v2]:
            complete_correct_trials += 1
        else:
            complete_incorrect_trials += 1
    print "Out of %d trials:" % trial_count
    print "\t %d returned a true partition" % complete_correct_trials
    print "\t %d returned a false partition" % complete_incorrect_trials
    print "\t %d rejected because of inconsistent signs" % rejected_inconsistent_trials
    print "\t %d rejected because it correctly thought the nodes were on the same side" % rejected_same_side_correct_trials
    print "\t %d rejected because it incorrectly thought the nodes were on the same side" % rejected_same_side_incorrect_trials

anchor_by_partition()

# TODO: once we get these anchor selection methods working, loop through all the nodes
# and decide which anchor to attach it with.

# TODO: for each other node w, assign it to maximize sign_stat(anchor(community), w)
# TODO: actually, just check whether N' is below "average" or not...?



