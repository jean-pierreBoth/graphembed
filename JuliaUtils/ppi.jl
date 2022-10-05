# conversion of Matlab files used in node2vec to csv containing triplet


using MAT

using Logging
using Base.CoreLogging
using Printf

logger = ConsoleLogger(stdout, CoreLogging.Debug)

# This functions read the file HomoSapiensPPi-node2vec.mat as processed in the paper
# *Grover Leskovec node2vec: Scalable Feature Learning for Networks 2016*.
# The processed file can be retrieved at http://snap.stanford.edu/node2vec
# network is symetric, but the whole edges are dumped in file.
# dumps matrix of graph in file ppi-network.csv and labels in file ppi-labels.csv
function ppi_network() 
    vars = matread("/home/jpboth/Data/Graphs/PPI/Homo_sapiens.mat")
    network = vars["network"]
    #
    networkio = open("ppi-network.csv","w")
    m = network.m
    n = network.n
    @info m,n
    nbloop = 0
    len = length(network.nzval)
    # network-ppi symetric graph all edges in file
    for j in 1:length(network.colptr) -1
        for l in network.colptr[j]:network.colptr[j+1]-1
            i = network.rowval[l]
            val = network.nzval[i]
            # we have a triplet i,j,val
            if i == j 
                nbloop += 1
            end
            @printf(networkio,"%d %d %d\n", i,j,val)
        end
    end
    @info "number of self loops" nbloop
    close(networkio)
    # dump labels
    group = vars["group"]
    labelsio = open("ppi-labels.csv","w")
    m = group.m
    n = group.n
    @info m,n
    for j in 1:length(group.colptr) -1
        for l in group.colptr[j]:group.colptr[j+1]-1
            i = group.rowval[l]
            val = group.nzval[i]
            # in fact val is a garbage value
            @printf(labelsio,"%d %d\n", i,j)
        end
    end
    close(labelsio)
end

