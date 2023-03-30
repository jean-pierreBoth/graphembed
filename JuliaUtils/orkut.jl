# to compare orkut decomposition and communities

using Logging
using Base.CoreLogging
using Printf

logger = ConsoleLogger(stdout, CoreLogging.Debug)

import JSON

# file containing the output of stable decomposition of orkut graph
orkutstable = "/home/jpboth/Rust/graphembed/orkut-decomposition.json"

# file containing first 5000 orkut communiites 
orkutcom = "/home/jpboth/Data/Graphs/Orkut/com-orkut.top5000.cmty.txt"


# read orkut communities, com[1] will contain nodes list of the first community

function getCommunities(orkutcom::String)
    lines = readlines(orkutcom)
    coms = map(l ->  split(l, '\t'),lines)
    nbcom = length(coms)
    communities = Vector{Vector{Int64}}(undef, nbcom)
    for i=1:nbcom
        communities[i] = map(s -> parse(Int64,s),  coms[i])
        if length(communities[i]) == 0 
            @info("null community %d", i)
        end
    end
    return communities
end




"""
 StableDecomposition

 A Julia structure to reload the json file written by graphembed

 The nodes are numbered from 0 as they come from Rust. Calls to functions referring to nodes 
 must be done with numbered numbered from 0. The functions do the translations internally

"""
struct StableDecomposition
    # s[i] contains the num of block (numbered from 0!) to which it belongs.
    s :: Vector{Int64}
    # 
    # list of numblocks as given in s but sorted in increasing num :  s[index[1]] <= s[index[2]] <=
    index :: Vector{Int64}
    # dimensionned to number of blocks
    # block[1] begins at 1, block[2] begins at index[block_start[1]]
    block_start :: Vector{Int64}
    #
    StableDecomposition(dict :: Dict{String, Any}) = new(Vector{Int64}(dict["s"]), Vector{Int64}(dict["index"]), Vector{Int64}(dict["block_start"]) )
end


# return a stable decomposition from a json file
function readStableDecompositionJson(path :: String)
    stable_json = JSON.parsefile(path,  dicttype=Dict, inttype=Int64, use_mmap=true)
    return StableDecomposition(stable_json)
end



# returns the member of block of rank blocknum
function get_block(stable :: StableDecomposition, blocnum::Int64)
    blocnum = blocnum + 1
    first = stable.block_start[blocnum] + 1
    if blocnum < length(stable.block_start)
        last = stable.block_start[blocnum+1]-1
    else 
        last = length(stable.index)
    end
    #
    return sort(stable.index[first:last])
end


"""
    returns the number of nodes of the graph stable decomposition
"""
function get_nb_block(stable::StableDecomposition)
    return length(stable.block_start)
end


"""
return the block of a given graph node
"""
function get_node_block(stable::StableDecomposition, node :: Int64)
    # increment node and node block with 1 to switch to Julia indexation!
    return stable.s[node + 1]
end


function communityBlockRange(community :: Vector{Int64},  stable :: StableDecomposition)
    #
    blocks = Vector{Int64}()
    for node in community
        block = get_node_block(stable, node)
        push!(blocks, block)
    end
    map(node ->  get_node_block(stable, node), community)
    return sort(blocks)
end


