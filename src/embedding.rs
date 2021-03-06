//! Describes the Embedded vectors. 
//! 
//! Basic symetric embedded vectors are described by an Array2<F>
//! 
//! for Hope embedding F can be a floating point type f32 or f64. So the type F is described by the constraints :  
//!   F : Float + Lapack + Scalar  + ndarray::ScalarOperand + sprs::MulAcc....   
//! For Nodesketch embedded the type is usize.  
//! 
//! Each row corresponds to a node.
//! 
//! For an asymetric or directed graph, an asymetric Embedded distinguishes for each node, its target role from its source role.
//! So we need 2 matrices to represent the Embedded data.
//! 
//! 
//! We have 2 Embedded modes.
//! - Hope is described by the paper :
//!     *Asymetric Transitivity Preserving Graph Embedded 2016* 
//!     M. Ou, P Cui, J. Pei, Z. Zhang and W. Zhu.
//! 
//! - Nodesketch is described in the paper <https://dl.acm.org/doi/10.1145/3292500.3330951>
//! 


use ndarray::{Array2, ArrayView1};
use indexmap::IndexSet;

use crate::tools::edge::{IN,OUT};
use crate::tools::degrees::*;

/// to represent the distance in embedded space between 2 vectors
type Distance<F> = fn(&ArrayView1<F>, &ArrayView1<F>) -> f64;

#[derive(Debug)]
pub enum EmbeddingMode {
    Hope, 
    NodeSketch,
}



/// The Embedded trait. It defines the interface satisfied by embedded data.  
/// In our implementations the embedded data are stored in Array2 and embedded node
/// are identified by their rank.  
/// F is the type contained in embedded vectors
pub trait EmbeddedT<F> {
    /// returns true if Embedded is symetric
    fn is_symetric(&self) -> bool;
    /// get dimension of vectors of the Embedded
    fn get_dimension(&self) -> usize;
    /// get distance in embedded space from node1 to node2 (same as distance from node2 to node1 if graph is symetric)
    /// Nodes are identified by their their rak in embedded space
    fn get_noderank_distance(&self, node_rank1: usize, node_rank2 : usize) -> f64;
    /// the trait provides a function distance between embedded items
    fn get_vec_distance(&self, v1 : &ArrayView1<F>, v2: &ArrayView1<F>) -> f64;
    /// get number of nodes
    fn get_nb_nodes(&self) -> usize;
    /// get embedding of node of rank rank, and with tag.
    /// For a basic symetric embedding , tag is not taken into account.
    /// For embedding that has multiple embedding by node (example asysmetric embedding , the tag is used)    
    fn get_embedded_node(&self, node_rank: usize, _tag : u8) -> ArrayView1<F>;
} // end of trait



/// represent symetric Embedded data
pub struct Embedded<F> {
    /// array (n,d) with n number of data, d dimension of Embedded
    data: Array2<F>,
    /// distance between vectors in embedded space. helps to implement trait EmbeddedT<F>
    distance : fn(&ArrayView1<F>, &ArrayView1<F>) -> f64,
} // end of Embedded



impl<F> Embedded<F> {
    // fills embedded vectors with the appropriate distance function
    pub(crate) fn new(arr : Array2<F>, distance : Distance<F>)  -> Self {
        Embedded{data : arr, distance : distance}
    }

    /// get representation of nodes as sources
    pub fn get_embedded(&self) -> &Array2<F> {
        &self.data
    }

    pub fn get_distance(&self) ->  &fn(&ArrayView1<F>, &ArrayView1<F>) -> f64 {
        &self.distance
    }

    /// get embedding of node of rank rank, and with tag.
    /// For a basic symetric embedding , tag is not taken into account.
    /// For embedding that has multiple embedding by node (example asysmetric embedding , the tag is used)
    pub fn get_embedded_node(&self, node_rank: usize, _tag : u8) -> ArrayView1<F> {
        self.data.row(node_rank)
    }
}  // end of impl Embedded



impl<F> EmbeddedT<F> for Embedded<F> {

    fn is_symetric(&self) -> bool {
        return true;
    }


    /// get dimension of Embedded. (row size of Array)
    fn get_dimension(&self) -> usize {
        self.data.dim().1
    }

    /// computes the distance in embedded space between 2 vectors
    /// dimensions must be equal to Embedded dimension
    fn get_vec_distance(&self, data1 : &ArrayView1<F>, data2: &ArrayView1<F>) -> f64 {
        assert_eq!(data1.len(), self.get_dimension());
        (self.distance)(data1, data2)
    }

    /// get distance between nodes identified by their rank!
    /// get distance from node1 to node2 (different from distance between node2 to node1 if Graph is asymetric)
    fn get_noderank_distance(&self, node1: usize, node2 : usize) -> f64 {
        (self.distance)(&self.data.row(node1), &self.data.row(node2))
    }

    ///
    fn get_nb_nodes(&self) -> usize {
        self.data.dim().0
    }

    /// get embedding of node of rank rank, and with tag.
    /// For a basic symetric embedding , tag is not taken into account.
    /// For embedding that has multiple embedding by node (example asysmetric embedding , the tag is used)
    fn get_embedded_node(&self, node_rank: usize, _tag : u8) -> ArrayView1<F> {
        self.data.row(node_rank)
    }

} // end impl EmbeddedT<F>

//===============================================================


/// Asymetric Embedded data representation
pub struct EmbeddedAsym<F> {
    /// source node representation
    source : Array2<F>,
    /// target node representation
    target : Array2<F>,
    ///
    degrees : Option<Vec<Degree>>,
    /// distance
    distance : Distance<F>,
} // end of struct EmbeddedAsym


impl <F> EmbeddedAsym<F> {

    pub(crate) fn new(source : Array2<F>, target : Array2<F>, degrees : Option<Vec<Degree>>, distance : Distance<F>) -> Self {
        assert_eq!(source.dim().0, target.dim().0);
        assert_eq!(source.dim().1, target.dim().1);
        EmbeddedAsym{source, target, degrees, distance}
    }

    /// get representation of nodes as sources
    pub fn get_embedded_source(&self) -> &Array2<F> {
        &self.source
    }

    /// get representation of nodes as targets
    pub fn get_embedded_target(&self) -> &Array2<F> {
        &self.target
    }
 
} // end of impl block for EmbeddedAsym



impl<F>  EmbeddedT<F> for EmbeddedAsym<F> {

    fn is_symetric(&self) -> bool {
        return false;
    }

    /// get dimension of Embedded. (row size of Array)
    fn get_dimension(&self) -> usize {
        self.source.dim().1
    }

    /// get distance from data1 to data2
    fn get_vec_distance(&self, data1 : &ArrayView1<F>, data2: &ArrayView1<F>) -> f64 {
        (self.distance)(data1, data2)
    }

    /// In this interface nodes are identified by their rank in the embedding, not their original identity
    /// So ranks must be less than number of nodes.
    /// To get an interface with original nodes id,  use the Embedding structure wwhich has a mapping from node_id to node_rank
    /// get distance FROM source node_rank1 TO target node_rank2 if Embedded is asymetric, in symetric case there is no order) 
    fn get_noderank_distance(&self, node_rank1 : usize, node_rank2 : usize) -> f64 {
        let mut distances = Vec::<f64>::with_capacity(3);
        //
        if let Some(degrees) = &self.degrees {
            let dist_s = (self.distance)(&self.source.row(node_rank1), &self.source.row(node_rank2));
            distances.push(dist_s);

            let dist_t = (self.distance)(&self.target.row(node_rank1), &self.target.row(node_rank2));
            distances.push(dist_t);
            //
            let dist_t = (self.distance)(&self.source.row(node_rank1), &self.target.row(node_rank2));
            distances.push(dist_t);
            if distances.len() > 0 {
                let dist = distances.iter().sum::<f64>() / distances.len() as f64;
                return dist;
            }
            else {
                log::error!("degrees node rank1 : {} degree = {:?}, node rank2 : {}, degree = {:?}", node_rank1, degrees[node_rank1], 
                                        node_rank2, degrees[node_rank2]);
                log::error!("get_noderank_distance asymetric no distance computed");
                return 1.;
            }
        }
        else {
            let dist = (self.distance)(&self.source.row(node_rank1), &self.target.row(node_rank2));
            return dist;
        }

    } // end of get_noderank_distance

    /// get number of nodes embedded.
    fn get_nb_nodes(&self) -> usize {
        self.source.dim().0
    }

    /// get embedding of node of rank rank, and with tag.
    /// For a basic symetric embedding , tag is not taken into account.
    /// For embedding that has multiple embedding by node (example asysmetric embedding , the tag is used)
    fn get_embedded_node(&self, node_rank: usize, tag : u8) -> ArrayView1<F> {
        match tag {
            OUT => { // returns embedding vector corresponding to node as a source or beginning of edge
                return self.source.row(node_rank);
            }
            IN => { // returns embedding vector corresponding to node as a target or end of edge
                return self.target.row(node_rank); 
            }
            _ => { 
                    log::error!(" for asymetric embedding tag in get_embedded_node must be 0 or 1");
                    std::panic!("for asymetric embedding tag in get_embedded_node must be 0 or 1");
            }
        }
    }
} // end impl EmbeddedT<F>


//====================================================================================


/// The trait Embedder is something that has as output :  
/// something satisfying the trait EmbeddedT<F>, for example  EmbeddedAsym<usize> for nodesketch embedding.  
/// F is the type contained in embedded vectors , mostly f64, f32, usize.  
/// Useful just to make cross validation generic.
pub trait EmbedderT<F>  {
    type Output : EmbeddedT<F> ;
    ///
    fn embed(& mut self) -> Result< Self::Output, anyhow::Error>;
} // end of trait EmbedderT<F>

//==============================================================================

/// The structure collecting the result of the embedding process
/// 
/// - F the embedded vectors contains values of type F (mostmy f32, f64, usize ...)
/// 
/// - NodeId is the type representing nodes (most often an usize). It must
///     implement Hash and Eq to be indexed.
/// 
/// - nodeindexation : an IndexSet storing Node identifier (as in datafile) and associating it to a rank in Array representing embedded nodes
///                      given a node id we get its rank in Array using IndexSet::get_index_of
///                      given a rank we get original node id by using IndexSet::get_index. 
/// 
/// - embbeded : the embedded data of type EmbeddedData. At present time Embedded<F> or EmbeddedAsym<F>
pub struct Embedding<F, NodeId : std::hash::Hash + std::cmp::Eq,  EmbeddedData : EmbeddedT<F> >  {
    /// association of nodeid to a rank. To be made generic to not restrict node id to usize
    nodeindexation : IndexSet<NodeId>,
    ///
    embedded : EmbeddedData,
    ///
    mark : std::marker::PhantomData<F>,
}  // end of Embedding



impl <NodeId, EmbeddedData,F> Embedding<F, NodeId, EmbeddedData >  where  EmbeddedData: EmbeddedT<F> ,
         NodeId :  std::hash::Hash + std::cmp::Eq {
    /// Creates an embedding of a Graph given a structure implementing an embedding (NodeSketch, NodeSketchAsym or Hope)
    /// If success return 
    pub fn new(nodeindexation : IndexSet<NodeId>, embedder : &mut dyn EmbedderT<F, Output = EmbeddedData>) -> Result<Self, anyhow::Error > {
        let embedded_res = embedder.embed();
        if embedded_res.is_err() {
            log::error!("embedding failed");
            return Err(embedded_res.err().unwrap());
        }
        else {
            return Ok(Embedding{nodeindexation, embedded : embedded_res.unwrap(), mark : std::marker::PhantomData })
        }
    } // end of new


    /// to retrieve the indexation
    pub fn get_node_indexation(&self) -> &IndexSet<NodeId> {
        return &self.nodeindexation
    } // end of get_node_indexation

    /// retrives the embedding asked for
    pub fn get_embedded_data(&self) -> &EmbeddedData {
        return &self.embedded
    } // end of get_embedded_data


    /// get distance between nodes, given their original node id
    pub fn get_node_distance(&self, node1 : NodeId, node2 : NodeId) -> f64 {
        let rank1 = self.nodeindexation.get_index_of(&node1).unwrap();
        let rank2 = self.nodeindexation.get_index_of(&node2).unwrap();
        self.embedded.get_noderank_distance(rank1, rank2)
    } // get_noderank_distance


    /// get rank of a node_id. 
    pub fn get_node_rank(&self, node_id: NodeId) -> Option<usize>  {
        self.nodeindexation.get_index_of(&node_id)
    }

    /// get node_id given its rank in indexation (and matrix representation)
    pub fn get_node_id(&self, rank: usize) -> Option<&NodeId> {
        self.nodeindexation.get_index(rank)
    }

 } // end of impl Embedding