# Results

We give here some embedding times and AUC for the link prediction task.

We estimate AUC as described in:  
*Link Prediction in complex Networks : A survey             
Lü, Zhou. Physica 2011.*  

The AUC runs with a fraction of edges deleted.  
The real fraction of edge deleted can be smaller than asked for as we avoid disconnecting nodes
form which nothing can be learned. We ask for deleting 15% of edges. The real fraction of edges deleted
is given in the column *"ratio discarded"*.

Computation times are those for **one embedding** with the same parameters as the validation and not the time 
for the validation iterations.

All computations are done on a 8-core i7@2.3 Ghz laptop

## Hope embedding results

### Adamic Adar mode

The eigenvalue range give the range between higher and lower extracted singular value.

The colmun *svd* specify how the randomized approximation of the svd was done.

AUC is run with 5 successive runs and 5000 edges deleted by run


1. Symetric Graphs

|  graph     | nb nodes | nb edges   |  svd(rank/epsil)      | ratio discarded | eigenvalue range | AUC (link)|  time(s)  |
|  ------    |  ---     | -------    |    :-------:          |   :-------:     |   :------:       |  ----     | :-----:   |
| Gnutella8  | 6301     | 20777      | rank 20, nbiter 10    |     0.137       |   21.6 - 4.8     |    0.82   |  1.2      |
| Gnutella8  | 6301     | 20777      | rank 100, nbiter 10   |     0.137       |   21.6 - 2.5     |    0.71   |  1.6      |
| ca-AstroPh | 18772    | 396160     | rank 100              |     0.15        |   83.7 - 14.3    |    0.964  |  11       |
| ca-AstroPh | 18772    | 396160     | rank 20               |     0.15        |   83.7 - 33      |    0.93   |  3.5      |
| youtube    | 1134890  | 2987624    | maxrank 30            |     0.11        |   4270 - 218     |    0.60   |   490     |
| youtube    | 1134890  | 2987624    | maxrank 75, bkiter 10 |     0.11        |   4270 - 140     |    0.64   |  1210     |


2. Asymetric Graphs

## Sketching embedding results 

The decay coefficient is the factor of reduction of edge weight at each new edge traversed during exploration around a node. 

AUC is run with 5 successive runs.
5000 edges deleted by pass

1. Symetric Graphs

|  graph        | nb nodes | nb edges   | dimension   |   nb hops    |  decay      |  ratio discarded |  AUC      | time(s)   |
|  :---:        |  :---:   | :-------:  |  :-------:  |   :-------:  |  :-------:  |   :---------:    |  :----:   | :-----:   |
| Gnutella8     |  6301    |   20777    |  100        |    5         |    0.2      |   0.137          |  0.93     |           |
| Gnutella8     |  6301    |   20777    |  100        |    10        |    0.2      |   0.137          |  0.90     |           |
| Gnutella8     |  6301    |   20777    |  200        |    10        |    0.2      |   0.137          |  0.96     |           |
| ca-AstroPh    | 18772    |  396160    |  100        |    5         |    0.2      |   0.148          |  0.968    |           |
| ca-AstroPh    | 18772    |  396160    |  100        |    10        |    0.2      |   0.148          |  0.948    |           |
| youtube       | 1134890  | 2987624    |  100        |    5         |    0.2      |   0.119          |  0.96     |   21      |
| youtube       | 1134890  | 2987624    |  100        |    10        |    0.2      |   0.119          |  0.948    |   36      |
| youtube       | 1134890  | 2987624    |  200        |    10        |    0.2      |   0.119          |  0.974    |   73      |


1. Asymetric Graphs


10000 edges deleted by pass


|  graph             | nb nodes | nb edges   | dimension   |  nb AUC pass | nb hops   |  decay    |  ratio     |  AUC     | time(s)   |
|  :----------:      |  :---:   | :-------:  |  :-------:  |   :-------:  |  :-----:  |   :-----: |  :----:    | :-----:  | :-------: |
| wiki_vote          | 7115     |  103689    |   100       |     20       |    5      |    0.1    |   0.147    |  0.883   |   0.5     |
| wiki_vote          | 7115     |  103689    |   200       |     20       |    5      |    0.1    |   0.147    |  0.896   |   ~1      |
| wiki_vote          | 7115     |  103689    |   500       |     20       |    5      |    0.1    |   0.147    |  0.93    |   ~1.5    |
| mmunmun_twitter    | 465017   |  834797    |   500       |     20       |    5      |    0.1    |   0.085    |  0.78    |   73      |


The munmun_twitter file is more difficult with very asymetric nodes having for example a in degree of 2 but an out degree of 493.