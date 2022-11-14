//! This module implements embedding of directed/undirected graph with labels attached to nodes/edges.
//! 
//! It uses the same strategy as the *nodesketch* module, see [nodesketch](crate::nodesketch).  
//! We hash labels as they propagate through the the edges of the graph.
//! It is inspired by the Weisfeiler-Lehman algorithm used in Graph Kernel litterature.
//! 
//!
//! Some References on Graph Kernels are :  
//! 
//! - Graph Kernels : A survey. Nikolentzos-Siglidis-Vazirgiannis 2021
//     panorama of graphs kernels. links with  GNN. Examples et perfs GNN et Graph Kernels comparées.
//     Core Weisfeiler-Lehman and Optimal Assignement Vertex Histogram are OK on unlabeled  or discrete labeled node graph
//! 
//! - Graph Kernels : State of the art and futures challenges  Borgwart 2020.
//     Taxonomy of graph kernels according to directed/undirected edges continuous/discrete labelling of nodes or edges.
//         Message passing Kernels of Nikolentsos seems good and covers the directed/undirected graphs and labelled/continuous nodes.
//         (but not edges labelling)
//!
//!  
// The first paper : 
// - Shervashidze-Borgwardt Weisfeiler-Lehman Graph Kernels 2011
//         provides a framework for kernels on unlabeled and discrete labels.
//         sorting neighbours labels+ compression (hash) and h iterations.
//         complexity h* nb edges
//         provides a feature vector for each node, and the whole graph
//         local and global WL!
// 
// - Power Iterated Color Refinement  Kersting-Grohe 2014
//         Establishes a link between power iterated color algorithm and relaxed matricial optimization between adjacency matrices. 
//         Systeme de Hash avec nombre premiers!
//  
// - Faster Kernels for Graphs with continuous Attributes via Hashing. 2016
// - Weisfeiler and Lehman go sparse Morris Rattan Mutzel 2020
// - Global Weisfeiler Lehman Kernel Morris-Kersting 2017
// 
// - Graph invariant Kernels Orsini IJCAI 2015   
//     definit un kernel pour les attributs. Le kernel global est le kernel sur les attributs * poids 
//     dependant du kernel sur les sommets et d'une fonction d appariement des structures matchées sous graphes.
//     gere les attributs continus et montre l amelioration des perf avec que sans.
//     WL meilleur kernel sur les sommets. Kernel sur les patterns , lables discrets, à batir avec du hash (par 3.1)
// 
// 
// Comparison GNN with KGraph
// 
//  - How powerful are Graph Neural Networks Xu-Hu Leskovec 2019
//      GNN <= Weisfeiler-Lehman test


//pub mod mgraph;
//pub mod sketch;

/// Defines interface to petgraph.  
pub mod pgraph;

/// Sketching on top of petgraph.  
pub mod psketch;

/// Defines sketching parameters.
pub mod params;

/// Defines translations of labels and ranks between raw data from io and our structures in MgraphSketcher.
pub mod idmap;

/// some utilities to load data examples.  
mod exio;