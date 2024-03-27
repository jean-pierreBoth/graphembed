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

Computation times are wall clock times those for **one embedding** with the same parameters as the validation and not the time
for the whole validation iterations. Cpu times depends on the level of parallelism of each algorithm.
Timings are given for a 24-core (32 threads) i9 laptop with 64Gb memory

## Symetric Graphs size 

|   Graph    |   nodes   |    edges    |
| :--------: | :-------: | :---------: |
| Gnutella8  |   6301    |   20 777    |
| ca-AstroPh |   18772   |   396 160   |
|   Amazon   |  334 000  |   925 000   |
|    Dblp    |  317079   |  1 049 866  |
|  youtube   | 1 134 890 |  2 987 624  |
|   orkut    | 3 072 441 | 117 185 083 |

## Hope embedding results

### Adamic Adar mode

The eigenvalue range give the range between higher and lower extracted singular value.

The column *svd* specify how the randomized approximation of the svd was done.
- (1) means rank subcommand 
- (2) means precision subcommand
AUC is run with at least 5 runs (depending on time needed by embedding).

#### Symetric Graphs

| graph      |           svd           | ratio discarded | eigenvalue range | AUC (link) | time(s) |
| ---------- | :---------------------: | :-------------: | :--------------: | :--------: | :-----: |
| Gnutella8  | rank 20, nbiter 10 (1)  |      0.137      |    21.6 - 4.8    |    0.82    |   1.2   |
| Gnutella8  | rank 100, nbiter 10 (1) |      0.137      |    21.6 - 2.5    |    0.71    |   1.6   |
|            |                         |                 |                  |            |         |
| ca-AstroPh | rank 100, bkiter 3 (2)  |      0.15       |      34 - 3      |    0.93    |   <1    |
| ca-AstroPh | rank 100, nbiter 5 (1)  |      0.15       |      35 - 5      |   0.938    |   <1    |
|            |                         |                 |                  |            |         |
| amazon     | rank 200, nbiter 5 (1)  |      0.15       |    150 - 9.9     |    0.84    |   63    |
|            |                         |                 |                  |            |         |
| dblp       | rank 400, nbiter 5 (1)  |      0.19       |      44 - 8      |   0.926    |   190   |
|            |                         |                 |                  |            |         |
| youtube    | rank 75, bkiter 3  (2)  |      0.12       |    4270 - 140    |    0.64    |   834   |
| youtube    | rank 30, nbiter 5  (1)  |      0.12       |    4270 - 471    |    0.90    |  1150   |

The rank subcommand of Hope embedding is more efficient

Centric Auc for Amazon : 0.834, std-dev = 0.006. Centric Auc is within 3 $\sigma$.  
  correlation(degree, auc) = 0.008

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


### Sketching: Symetric Graphs


#### embedding results
  
|   graph    | dimension | nb hops | decay | ratio discarded |   AUC +- sigma   | time(s) |
| :--------: | :-------: | :-----: | :---: | :-------------: | :--------------: | :-----: |
| Gnutella8  |    500    |    5    |  0.2  |      0.137      |  0.804 +- 0.001  |   0.4   |
| Gnutella8  |    500    |    5    |  0.4  |      0.137      | 0.805 +- 0.0009  |   0.4   |
|            |           |         |       |                 |                  |         |
| ca-AstroPh |    100    |    5    |  0.2  |      0.148      |      0.947       |   0.5   |
| ca-AstroPh |    200    |    5    |  0.2  |      0.148      |  0.96 +- 0.0004  |   0.6   |
|            |           |         |       |                 |                  |         |
|    Dblp    |    100    |    5    |  0.5  |      0.19       | 0.9013 +- 6.6E-4 |    4    |
|    Dblp    |    400    |    4    |  0.4  |      0.19       | 0.9611 +- 4.4E-4 |  13.8   |
|            |           |         |       |                 |                  |         |
|   amazon   |    200    |    3    |  0.3  |      0.118      | 0.963 +- 2.3 E-4 |    5    |
|            |           |         |       |                 |                  |         |
|  youtube   |    200    |    2    |  0.2  |      0.119      |  0.914 +- 0.001  |   13    |
|  youtube   |    200    |    3    |  0.2  |      0.119      |  0.899 +- 0.002  |   18    |
|  youtube   |   1000    |    5    |  0.5  |      0.119      |  0.908 +- 0.002  |   145   |
|            |           |         |       |                 |                  |         |
|   orkut    |    200    |    3    |  0.2  |      0.149      | 0.924 +- 0.0008  |   240   |
|   orkut    |    200    |    5    |  0.2  |      0.149      | 0.948 +- 0.0011  |   260   |
|   orkut    |    200    |    5    |  0.3  |      0.149      | 0.953 +- 0.0007  |   260   |
|   orkut    |    300    |    5    |  0.4  |      0.149      |  0.96 +- 0.0004  |   260   |



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
