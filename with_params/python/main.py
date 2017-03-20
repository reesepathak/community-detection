from util import *
from read_data import *
import networkx as nx
import pdb

G = graph_from_file("../data/polblogs.gml")
left, right = subgraphs_by_value(G, [0, 1])
p = [get_conn_prob(left), get_conn_prob(right)]
pdb.set_trace()