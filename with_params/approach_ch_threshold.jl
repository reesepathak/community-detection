# approach_ch_threshold.jl
# For Mark's data collection purposes

include("helpers.jl")
using LightGraphs, PyPlot
using HDF5
include("sbm.jl")

function main(n, k, a, b)
    nstr = string(n); kstr = string(k); astr = string(a); bstr = string(b);
    println("Generating SBM... n = $nstr, k = $kstr, a = $astr, b = $bstr")
    labels, edgeSet = gen_sym_sbm(n, k, a, b, regime=SBM_LOG)
    original, true_classes = to_lightgraph(labels, edgeSet)

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

    numTrials = 3
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

    # outfile = "threshold/results_$(astr)_$(bstr).h5"
    # h5write(outfile, "accuracies_$(astr)_$(bstr)", accuracies)
    # h5write(outfile, "cluster_sizes_$(astr)_$(bstr)", cluster_sizes)
    # h5write(outfile, "classifications_$(astr)_$(bstr)", classifications)

    # Return the average accuracy:
    return mean(accuracies)
end
amin = 1; amax = 10
bmin = 0.02; bmax = 1
avec = linspace(amin, amax, 40)
bvec = linspace(bmin, bmax, 40)
n = 300
k = 2
accuracygrid = zeros(length(avec), length(bvec))
for i in 1:length(avec)
    for j in 1:length(bvec)
        accuracygrid[i,j] = main(n, k, avec[i], bvec[j])
    end
end
println("Writing to file...")
h5write("threshold_$(amin)_$(amax)_$(bmin)_$(bmax).h5", "accuracies", accuracygrid)
# accuracygrid = h5read("threshold.h5", "accuracies")

# Plot Output
xs = [string(elem) for elem in avec]
ys = [string(elem) for elem in bvec]
z = accuracygrid

# Plot
fig, ax = subplots()
# Accuracy
pcolormesh(bvec, avec, accuracygrid, cmap=ColorMap("Blues"))

# Superimpose threshold
thresh = (sqrt(2) + sqrt(bvec)).^2
plot(bvec, thresh, color = "red", linestyle="-")
xlim([bmin, bmax])
ylim([amin, amax])
xlabel("b")
ylabel("a")
colorbar()
title("Plot of recovery accuracy for various values of a, b")
savefig("accuracy.png")
