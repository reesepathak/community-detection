include("helpers.jl")
using LightGraphs, HDF5
include("sbm.jl")

FNAME = "data/polblogs.gml"
println("Loading Political Blog Dataset ....")
original = loadgraph(FNAME, :gml)
true_classes = return_classtable(FNAME)

# println("Generating SBM...")
# labels, edgeSet = gen_sym_sbm(1200, 2, 5.0, 4.0, regime=SBM_LOG)
# original, true_classes = to_lightgraph(labels, edgeSet)

println("Finding Largest Connected Component")
# ccs = weakly_connected_components(original)
ccs = connected_components(original)
largest_cc = ccs[indmax(map(x->size(x)[1], ccs))]
println("Obtaining Subgraph for Largest Connected Component")
blog_dataset, _ = induced_subgraph(original, largest_cc)
true_classes = true_classes[largest_cc]
blog_dataset = Graph(blog_dataset)
num_blogs = nv(blog_dataset)
true_cluster_sizes_1 = sum(true_classes)
true_cluster_sizes_2 = length(true_classes) - true_cluster_sizes_1
println("Ground Truth: Cluster sizes are $true_cluster_sizes_1 and $true_cluster_sizes_2")
println("Obtaining a classification of nodes")
meanDegree = sum(adjacency_matrix(blog_dataset))/(nv(blog_dataset))
println("Mean degree is $meanDegree")

numTrials = 40
cluster_sizes = []
accuracies = []
classifications = []
for i=1:numTrials
    classification = repeater(blog_dataset, meanDegree)
    acc_rate_same = sum(1.0*(classification .== true_classes))/num_blogs
    acc_rate_swap = sum(1.0*(classification .!= true_classes))/num_blogs
    acc_rate_final = max(acc_rate_same, acc_rate_swap)
    cluster_sizes_1 = sum(classification)
    cluster_sizes_2 = length(classification) - cluster_sizes_1
    push!(cluster_sizes, [cluster_sizes_1, cluster_sizes_2])
    push!(accuracies, acc_rate_final)
    push!(classifications, classification)
    print("Classification $i. Cluster 1: $cluster_sizes_1. Cluster 2: $cluster_sizes_2.") 
    println("...\t Accuracy: $acc_rate_final")
end

outfile = "results.h5"
h5write(outfile, "accuracies", accuracies)
h5write(outfile, "cluster_sizes", cluster_sizes)
h5write(outfile, "classifications", classifications)

