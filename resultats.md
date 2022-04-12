# resultats

We give here some embedding times and AUC for the link prediction task
All computations are done on a 8-core i7@2.3 Ghz

## Results Hope embedding

The eigenvalue range gives the range between higher and lower extracted singular values 

1. Symetric Graphs

|  graph     | nb nodes | nb edges   | hope mode   |  svd(rank/epsil)  | ratio discarded | eigenvalue range | AUC (link)|  time(s)  |
|  ---       |  ---     | -------    |  -------    |    -------        |   -------       |   ------         |  ----     | -----     |
| Gnutella8  | 6301     | 20777      |  ADA        |rank 20, nbiter 10 |     0.137       |   21.6 - 4.8     |    0.82   |  1.2      |
| Gnutella8  | 6301     | 20777      |  ADA        |rank 100, nbiter 10|     0.137       |   21.6 - 2.5     |    0.71   |  1.6      |
| ca-AstroPh | 18772    | 396160     |  ADA        |   rank 100        |     0.15        |   83.7 - 14.3    |    0.964  |  11       |
| ca-AstroPh | 18772    | 396160     |  ADA        |   rank 20         |     0.15        |   83.7 - 33      |    0.93   |  3.5      |


2. Asymetric Graphs

## Results Sketching 

1. Symetric Graphs

|  graph     | nb nodes | nb edges   | dimension   |   nb hops    |  decay      |  ratio discarded |  AUC      | time(s)   |
|  :---:     |  :---:   | :-------:  |  :-------:  |   :-------:  |  :-------:  |   :---------:    |  ----     | :-----:   |
| Gnutella8  |  6301    |   20777    |  100        |    5         |    0.2      |   0.137          |  0.93     |           |
| Gnutella8  |  6301    |   20777    |  100        |    10        |    0.2      |   0.137          |  0.90     |           |
| Gnutella8  |  6301    |   20777    |  200        |    10        |    0.2      |   0.137          |  0.96     |           |
| ca-AstroPh | 18772    |  396160    |  100        |    5         |    0.2      |   0.148          |  0.968    |           |
| ca-AstroPh | 18772    |  396160    |  100        |    10        |    0.2      |   0.148          |  0.948    |           |
| youtube    | 1134890  | 2987624    |  100        |    5         |    0.2      |   0.119          |  0.96     |   21      |
| youtube    | 1134890  | 2987624    |  100        |    10        |    0.2      |   0.119          |  0.948    |   36      |
| youtube    | 1134890  | 2987624    |  200        |    10        |    0.2      |   0.119          |  0.974    |   73      |


2. Asymetric Graphs