# Orkut embedded graph density analysis

We examine the distances between neighbours in the ANN graph in the embedding space.

We give here the distribution of the ratio of mean distance between embedded neighbours inside a commmon block
to the mean distance between neighbours not in the same block. The quantiles are taken on the 208 blocks of the decomposition.

|  quantile  | d_in/d_out |
|  :---:     |  :---:     |
|   0.05     |  0.48      |
|   0.1      |  0.52      |
|   0.25     |  0.67      |
|   0.5      |  0.82      |  
|   0.75     |  0.87      |
|   0.9      |  0.89      |
|   0.95     |  0.90      |

The mean ratio weighted by the total degree of departure block is 0.81

The embedded distance between neighbours in the same block is consistently less than the distance between neighbours not in the same blocks.
Moreover the mean ratio obtained by weighting the ratio by the total degree of a block (number of edges with at least one point in a given block) is a bit less than the median so larger block seem also consistently embedded.

## Community edges analysis

As we have an experimental community decomposition of the graph we can do the same analysis as above with communities instead of blocks.
We used the 5000 first communities.

|  quantile  | d_in/d_out |
|  :---:     |  :---:     |
|   0.05     |  0.40      |
|   0.25     |  0.53      |
|   0.5      |  0.626     |  
|   0.75     |  0.754     |
|   0.90     |  0.974     |
|   0.95     |  1.09      |

- The mean ratio is 0.669
- The mean ratio, when weighted by community size is 0.697. This shows that there is no degradation of distances
  inside smaller communities, crossing community boundary is more penalized for small communities than large ones.
- There are 433 (over 5000) communities with a mean internal edge length  **greater** than the mean length of edges going out the community.  
  Among them some very small communities for example 1493, 3719 and 4760 of respective sizes (3, 4 and 3)
  have internal edges more 2.5 greater than edges crossing boundary.

## Block transition analysis

For a given block, edges having one node in a block have a neighbour in some arrival block (possibly the same block).
We record this arrival distribution for each block and compute the kl_divergene between transitions
observed in the original graph and the approximate neighbourhood graph (which is asymetric) deduced from the embedded data.  

Statistics are gathered on the 208 blocks.

|  quantile  | blocks kld |
|  :---:     |  :---:     |
|   0.05     |  0.044     |
|   0.1      |  0.046     |
|   0.25     |  0.056     |
|   0.5      |  0.070     |  
|   0.75     |  0.086     |
|   0.9      |  0.096     |
|   0.95     |  0.110     |

Transition from blocks are in agreement between the original graph and the embedded.  

In fact we observe an increase in kl-divergence towards the low density blocks.
Transitions from block 207 has a divergence of 1.1 between original graph and the embedded one.
