# Graphembed

The purpose of this crate is to provide asymetric (and also symetric) embedding of graphs positively weighted edges.
The crate comes as a library and as a simple executable with a validation option.  
It is still in an early version.
## Methods

We use two strategies for graph embedding.
1. The first is based on the paper : 

*NodeSketch : Highly-Efficient Graph Embeddings via Recursive Sketching KDD 2019*.  [https://dl.acm.org/doi/10.1145/3292500.3330951]  
    D. Yang,P. Rosso,Bin-Li, P. Cudre-Mauroux. 

It is based on multi hop neighbourhood identification via sensitive hashing.  
Instead of using **ICWS** for hashing we use the more recent algorithm **probminhash**. See [probminhash](https://arxiv.org/abs/1911.00675).
The algorithm associates a probability distribution on neighbours of each point depending on edge weights and distance to the point.
Then this distribution is hashed to build an embedding vector. The distance between embedded vectors is the Jaccard distance so we get
a real distance on the embedding space, at least for the symetric embedding.
An extension of the paper is also implemented to get asymetric embedding for directed graph. The similarity is also based on the hash of sets (nodes going to or from) a given node but then the dissimilarity is no more a distance (no symetry and some discrepancy with the triangular inequality).

2. The second is based on the paper:
   
*Asymetric Transitivity Preserving Graph Embedding 2016*  
    M. Ou, P Cui, J. Pei, Z. Zhang and W. Zhu.

The objective is to provide an asymetric graph embedding and get estimate of the precision of the embedding in function of its dimension.
We use the Adamic-Adar matricial representation of the graph. (It must be noted that the ponderation of a node by the number of couples joined by it is called Resource Allocation in the Graph Kernel litterature).
The asymetric embedding is obtained from the left and right singular eigenvectors of the Adamic-Adar representation of the graph.
Source node are related to left singular vectors and target nodes to the right ones. The similarity measure is the dot product, so it is not a norm.  
The svd is approximated by randomization as described in Halko-Tropp 2011 as implmented in the annembed crate.




## Some data sets



Small datasets are given in the Data subdirectory. Larger datasets can 
be downloaded from the SNAP data collections <https://snap.stanford.edu/data>



#### Some small test graphs are provided in a Data subdirectory.


1. Symetric graphs 

* Les miserables  <http://konect.cc/>   
    les miserables  co occurence de mots dans un chapitre

* CA-GrQc.txt       <https://snap.stanford.edu/data/ca-GrQc.html>

*   p2p-Gnutella08.txt.gz   <https://snap.stanford.edu/data/p2p-Gnutella08.html>

1. Asymetric graphs
   
*   wiki-vote               <https://snap.stanford.edu/data/wiki-Vote.html>
        7115 nodes 103689 edges
    
*   soc-epinions            <https://snap.stanford.edu/data/soc-Epinions1.html>
        8114 nodes, 26013 edges
   
* Cora : <http://konect.cc/networks/subelj_cora>
        citation network 23166 nodes 91500 edges

#### Some larger data tests for user to download

These graphs were used in results see below.

Beware of the possible need to convert from Windows to Linux End Of Line, see the dos2unix utility.  
Possibly some data can need to be converted from Tsv format to Csv, before being read by the program. 

1. Symetric 

* youtube.  Nodes: 1134890 Edges: 2987624 <https://snap.stanford.edu/data/com-Youtube.html>

2. Asymetric
   
* twitter as tested in Hope  <http://konect.cc/networks/munmun_twitter_social>
        465017 nodes 834797 edges


## Some results

Embedding and link prediction evaluation for the above data sets are given [here](./resultats.md)

### Some qualitative comments

## Installation and Usage

### Installation

The crate provides two features, required by the *annembed* dependency, to specify which version of lapack you want to use.  
For example compilation is done by :
*cargo build --release --features="openblas-static"* to link statically with openblas.
The choice of one feature is mandatory to provide required linear algebra library.
### Usage

* The Hope embedding relying on matrices computations limits the size of the graph to some hundred thousands nodes.
It is intrinsically asymetric in nature. It nevertheless gives access to the spectrum of Adamic Adar representing the graph and
so to the required dimension to get a valid embedding in $R^{n}$.  
The Sketching embedding is much faster for large graphs but embeds in a space consisting in sequences of node id equipped with the Jaccard distance.

* The *embed* module takes embedding and possibly validation commands in one directive.  
The general syntax is :

    embed file_description [validation_command --validation_arguments] embedding_command --embedding_arguments

    It is detailed in docs of the embed module. Use cargo doc --no-deps as usual.

* Use the environment variable RUST_LOG gives access to some information at various level (debug, info, error)  via the **log** and **env_logger** crates.