# Graphite

The purpose of this crate is to provide asymetric (and also symetric) embedding of graphs

**work in progress...**
## Methods

We use two strategies for graph embedding.
1. The first is based on the paper : 

*NodeSketch : Highly-Efficient Graph Embeddings via Recursive Sketching KDD 2019*.  [https://dl.acm.org/doi/10.1145/3292500.3330951]  
    D. Yang,P. Rosso,Bin-Li, P. Cudre-Mauroux. 

It is based on multi hop neighbourhood identification via sensitive hashing.  
Instead of using **ICWS** for hashing we use the more recent algorithm **probminhash** See [probminhash](https://arxiv.org/abs/1911.00675).
The idea is to associate a probability distribution of neighbours of each point and hash this distribution to build an embedding vector.
An extension of the paper is also implemented to get asymetric embedding for directed graph. In fact by default the algorithm runs in
asymetric mode.

2. The second is based on the paper:
   
*Asymetric Transitivity Preserving Graph Embedding 2016*  
    M. Ou, P Cui, J. Pei, Z. Zhang and W. Zhu.

The objective is to provide an asymetric graph embedding and get estimate of the precision of the embedding in function of its dimension.
We use the Adamic-Adar (also known as Resource Allocatior in Kernel Graph litterature) representation of the graph.
The asymetric embedding is obtined from the left and right singular eigenvectors of thr Adamic-Adar representatino of the graph.  
The svd is approximated by randomization as described in Halko-Tropp 2011. 

Katz index or Rooted Page Rank should also be possible using randomized Gsvd as described in :
 *Randomized Generalized Singular Value Decomposition CAMC 2021*
    W. Wei H. Zhang, X. Yang, X. Chen

## Some results

The results are detailed [here](./resultats.md)

## Some data sets



Small datasets are given in the Data subdirectory. Larger datasets can 
be downloaded from the SNAP data collections <https://snap.stanford.edu/data>



#### Some small test graphs are provided in a Data subdirectory.


1. Symetric graphs 

* Les miserables  <http://konect.cc/>   
    les miserables  co occurence de mots dans un chapitre

* CA-GrQc.txt       <https://snap.stanford.edu/data/ca-GrQc.html>

2. Asymetric graphs
   
*   wiki-vote               <https://snap.stanford.edu/data/wiki-Vote.html>
*   soc-epinions            <https://snap.stanford.edu/data/soc-Epinions1.html>
*   p2p-Gnutella09.txt.gz   <https://snap.stanford.edu/data/p2p-Gnutella09.html>
        8114 nodes, 26013 edges
   
* Cora : <http://konect.cc/networks/subelj_cora>
        citation network 23166 nodes 91500 edges

#### Some larger data tests for user to download

1. Symetric 

* youtube.  Nodes: 1134890 Edges: 2987624 <https://snap.stanford.edu/data/com-Youtube.html>

2. Asymetric
   
* twitter as tested in Hope  <http://konect.cc/networks/munmun_twitter_social>
        465017 nodes