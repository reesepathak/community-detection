# spectral_comm_detect.jl
# Spectral Community Detection
# EE 376A


using LightGraphs, PyPlot
include("sbm.jl")

# function partition_cproj()
    # ;
# end

"""
    partition_2evec(Graph)

    Use the 2nd eigenvector of the graph Laplacian 
    to detect communities in the graph.
"""
function partition_2evec(G)
    # Grab 2nd eigenvector
    e2 = eigvecs(full(laplacian_matrix(G)))[:,2]

    plot(sorted(e2))

end


labels,edgeSet = gen_sym_sbm(100, 2, 5, 1, regime=SBM_LOG)
G, labels = to_lightgraph(labels, edgeSet)




