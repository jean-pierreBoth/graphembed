# GraphEmbed

**work in progress...**
## Methods

We use two strategies for graph embedding.
1. The first is based on the paper : 

*NodeSketch : Highly-Efficient Graph Embeddings via Recursive Sketching KDD 2019*.  [https://dl.acm.org/doi/10.1145/3292500.3330951]  
    D. Yang,P. Rosso,Bin-Li, P. Cudre-Mauroux. 

It is based on multi hop neighbourhood identification via sensitive hashing. But instead of using **ICWS** for hashing we use the more recent **probminhash** See [probminhash](https://arxiv.org/abs/1911.00675)
An extension of the paper is also implemented to get asymetric embedding for directed graph.

2. The second is based on the paper:
   
*Asymetric Transitivity Preserving Graph Embedding 2016*  
    M. Ou, P Cui, J. Pei, Z. Zhang and W. Zhu.

The main idea of the paper is to preserve multi-hop proximity and constructing an asymetric graph embedding. 
It relies on Generalized SVD. The approximation of the generalized we used is based on randomization 
as described in 
*Randomized General Singular Value Decomposition CAMC 2021*
    W. Wei H. Zhang, X. Yang, X. Chen


## Some data sets

1. Symetric graphs 

* Les miserables  http://konect.cc/   
    les miserables  co occurence de mots dans un chapitre

* Cora : http://konect.cc/networks/subelj_cora
    citation network 23166 neoeuds 91500 aretes  no loop  dout max 104 d in max = 376 (weighted)


2. Asymetric graphs
   
* CA-GrQc.txt       <https://snap.stanford.edu/data/ca-GrQc.html>
*   wiki-vote       <https://snap.stanford.edu/data/wiki-Vote.html>
*   soc-epinions    <https://snap.stanford.edu/data/soc-Epinions1.html>
   
