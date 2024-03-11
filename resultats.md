# Results

We give here some embedding times and AUC for the link prediction task.

We estimate AUC as described in:  
*Link Prediction in complex Networks : A survey  
Lü, Zhou. Physica 2011.*  

The AUC runs with a fraction of edges deleted.  
The real fraction of edge deleted can be smaller than asked for as we avoid disconnecting nodes
form which nothing can be learned. We ask for deleting 15% of edges. The real fraction of edges deleted
is given in the column *"ratio discarded"*.

Edges of symetric graphs are treated as 2 different edges, one for each orientation. For link prediction, in the symetric case the 2 orientated
edges are deleted,  in asymetric case, only one edge is deleted.

Computation times are those for **one embedding** with the same parameters as the validation and not the time
for the whole validation iterations.

Timings are given for a 24-core (32 threads) i9 laptop with 64Gb memory
## Hope embedding results

### Adamic Adar mode

The eigenvalue range give the range between higher and lower extracted singular value.

The column *svd* specify how the randomized approximation of the svd was done.

AUC is run with 5 successive runs.

#### Symetric Graphs

| graph      | nb nodes | nb edges |    svd(rank/epsil)    | ratio discarded | eigenvalue range | AUC (link) | time(s) |
| ---------- | -------- | -------- | :-------------------: | :-------------: | :--------------: | ---------- | :-----: |
| Gnutella8  | 6301     | 20777    |  rank 20, nbiter 10   |      0.137      |    21.6 - 4.8    | 0.82       |   1.2   |
| Gnutella8  | 6301     | 20777    |  rank 100, nbiter 10  |      0.137      |    21.6 - 2.5    | 0.71       |   1.6   |
| ca-AstroPh | 18772    | 396160   |       rank 100        |      0.15       |   83.7 - 14.3    | 0.964      |   11    |
| ca-AstroPh | 18772    | 396160   |        rank 20        |      0.15       |    83.7 - 33     | 0.93       |   3.5   |
| youtube    | 1134890  | 2987624  |      maxrank 30       |      0.11       |    4270 - 218    | 0.60       |   490   |
| youtube    | 1134890  | 2987624  | maxrank 75, bkiter 10 |      0.11       |    4270 - 140    | 0.64       |  1210   |

#### Asymetric Graphs

AUC is estimated on 10 passes.

| graph | nb nodes | nb edges |    svd(rank/epsil)     | ratio discarded | eigenvalue range | AUC (link) | time(s) |
| :---: | :------: | :------: | :--------------------: | :-------------: | :--------------: | ---------- | :-----: |
| Cora  |  23166   |  91500   | maxrank 50, bkiter 10  |      0.144      |    ~ 6. - 1.     | 0.81       |   0.3   |
| Cora  |  23166   |  91500   | maxrank 200, bkiter 10 |      0.144      |    ~ 7. - 0.8    | 0.837      |   1.7   |
| Cora  |  23166   |  91500   |  rank 200, bkiter 10   |      0.144      |   ~ 7.5 - 1.5    | 0.86       |   5.9   |
| Cora  |  23166   |  91500   |  rank 400, bkiter 10   |      0.144      |   ~ 7.5 - 1.1    | 0.84       |  14.4   |

## Sketching embedding results

The decay coefficient is the factor of reduction of edge weight at each new edge traversed during exploration around a node.

AUC is run with 5 successive runs.

### Sketching: Symetric Graphs

#### Gnutella8 graph: nb nodes 6301, nb edges 20777
  
| dimension | nb hops | decay | ratio discarded |  AUC +- sigma   | time(s) |
| :-------: | :-----: | :---: | :-------------: | :-------------: | :-----: |
|    200    |    3    |  0.2  |      0.137      | 0.792 +- 0.001  |   0.1   |
|    500    |    5    |  0.2  |      0.137      | 0.804 +- 0.001  |   0.4   |
|    500    |    5    |  0.4  |      0.137      | 0.805 +- 0.0009 |   0.4   |

#### ca-AstroPh graph: nb nodes 18772, nb edges 396160
 
| dimension | nb hops | decay | ratio discarded |  AUC +- sigma  | time(s) |
| :-------: | :-----: | :---: | :-------------: | :------------: | :-----: |
|    100    |    5    |  0.2  |      0.148      |     0.947      |   0.5   |
|    200    |    5    |  0.2  |      0.148      | 0.96 +- 0.0004 |   0.6   |


#### youtube graph: nb nodes 1 134 890,   nb edges : 2 987 624 

| dimension | nb hops | decay | ratio discarded |  AUC +- sigma  | time(s) |
| :-------: | :-----: | :---: | :-------------: | :------------: | :-----: |
|    200    |    2    |  0.2  |      0.119      | 0.914 +- 0.001 |   13    |
|    200    |    3    |  0.2  |      0.119      | 0.901 +- 0.001 |   18    |

#### orkut graph. nb nodes 3 072 441 nb edges : 117 185 083

| dimension | nb hops | decay | ratio discarded |  AUC +- sigma   | time(s) |
| :-------: | :-----: | :---: | :-------------: | :-------------: | :-----: |
|    200    |    3    |  0.2  |      0.149      | 0.924 +- 0.0008 |   240   |
|    200    |    5    |  0.2  |      0.149      | 0.948 +- 0.0011 |   260   |
|    200    |    5    |  0.3  |      0.149      | 0.953 +- 0.0007 |   260   |
|    300    |    5    |  0.4  |      0.149      | 0.96 +- 0.0004  |   260   |

### Sketching: Asymetric Graphs

standard deviation on AUC is around 8.E-4 with 20 AUC pass

#### wiki vote graph nb nodes : 7115, nb edges : 103689

| dimension | nb AUC pass | nb hops | decay | ratio |  AUC  | time(s) |
| :-------: | :---------: | :-----: | :---: | :---: | :---: | :-----: |
|    100    |     20      |    5    |  0.1  | 0.147 | 0.883 |   0.5   |
|    200    |     20      |    5    |  0.1  | 0.147 | 0.896 |   ~1    |
|    500    |     20      |    5    |  0.1  | 0.147 | 0.922 |  ~1.5   |
|    500    |     20      |    2    | 0.25  | 0.147 | 0.94  |  ~1.5   |


#### cora graph.  nb nodes : 7115, nb edges : 103689

| dimension | nb AUC pass | nb hops | decay | ratio |  AUC  | time(s) |
| :-------: | :---------: | :-----: | :---: | :---: | :---: | :-----: |
|    200    |     20      |    5    |  0.2  | 0.143 | 0.924 |   ~1.   |
|    300    |     40      |    5    |  0.5  | 0.143 | 0.932 |   ~2.   |


#### mmunmun_twitter graph. nb nodes : 465017, nb edges : 834797

| dimension | nb AUC pass | nb hops | decay | ratio |  AUC  | time(s) |
| :-------: | :---------: | :-----: | :---: | :---: | :---: | :-----: |
|    500    |     20      |    5    |  0.1  | 0.085 | 0.78  |   73    |
|    500    |     20      |    5    |  0.2  | 0.085 | 0.787 |   74    |
|   1000    |     20      |    5    |  0.2  | 0.085 | 0.80  |   160   |
|   1000    |     20      |    5    |  0.5  | 0.085 | 0.788 |   160   |

The munmun_twitter graph has characteristics,known in the litterature dedicated to large sparse directed graphs, that make it difficult:

- very asymetric nodes having for example in degree of 2 but an out degree more than 450

- low mean degree.

- huge imbalance between a small number of existing directed edges (less than $10^{6}$ ) and a much larger set of potential edges (here more than $4 \cdot 10^{11}$).  

In fact there are only 1257 edges that are bidirectional in the munmun graph so we can measure the impact of symetrization and make all edges bidrectional.
Keeping decay at 0.2, nb_hop = 5, we get AUC = 0.91 with dim = 500 and AUC = 0.942 with dim = 1000.
