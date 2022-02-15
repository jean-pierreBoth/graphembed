//! Describes the embedding vectors. 
//! 
//! Basic symetric Embedding are described by an Array2<F>
//! 
//! F can be a floating point type f32 or f64, in this case thetype F is described by the constraints :  
//!   F : Float + Lapack + Scalar  + ndarray::ScalarOperand + sprs::MulAcc.... 
//! 
//! For Nodesketch embedding the type is usize.  
//! Each row corresponds to a node.
//! 
//! For an asymetric or directed graph, an asymetric embedding distinguishes for each node, its target role from its source role.
//! So we need 2 matrices to represent the embedding.
//! 
//! 



use ndarray::{Array2, ArrayView1};

type Distance<F> = fn(&ArrayView1<F>, &ArrayView1<F>) -> f64;


pub enum EmbeddingMode {
    Hope, 
    Nodesketch,
}



pub trait EmbeddingT<F> {
    /// returns true if embedding is symetric
    fn is_symetric(&self) -> bool;
    /// the trait provides a function distance between embedded items
    fn get_vec_distance(&self, v1 : &ArrayView1<F>, v2: &ArrayView1<F>) -> f64;
    ///
    fn get_dimension(&self) -> usize;
    /// get distance from node1 to node2 (same as distance from node2 to node1 if graph is symetric)
    fn get_node_distance(&self, node1: usize, node2 : usize) -> f64;
} // end of trait



/// symetric embedding
pub struct Embedding<F> {
    /// array (n,d) with n number of data, d dimension of embedding
    data: Array2<F>,
    /// distance
    distance : fn(&ArrayView1<F>, &ArrayView1<F>) -> f64,
} // end of Embedding



impl<F> Embedding<F> {
    // fills embedded vectors with the appropriate distance function
    pub(crate) fn new(arr : Array2<F>, distance : Distance<F>)  -> Self {
        Embedding{data : arr, distance : distance}
    }

}  // end of impl Embedding



impl<F> EmbeddingT<F> for Embedding<F> {

    fn is_symetric(&self) -> bool {
        return true;
    }

    /// get dimension of embedding. (row size of Array)
    fn get_dimension(&self) -> usize {
        self.data.dim().1
    }

    /// computes the distance in embedded space between 2 vectors
    /// dimensions must be equal to embedding dimension
    fn get_vec_distance(&self, data1 : &ArrayView1<F>, data2: &ArrayView1<F>) -> f64 {
        assert_eq!(data1.len(), self.get_dimension());
        (self.distance)(data1, data2)
    }

    /// get distance from node1 to node2 (different from distance between node2 to node1 if embedding is asymetric)
    fn get_node_distance(&self, node1: usize, node2 : usize) -> f64 {
        (self.distance)(&self.data.row(node1), &self.data.row(node2))
    }

} // end impl EmbeddingT<F>

//===============================================================


/// Asymetric embedding representation
pub struct EmbeddingAsym<F> {
    /// source node representation
    source : Array2<F>,
    /// target node representation
    target : Array2<F>,
    /// distance
    distance : Distance<F>,
} // end of struct EmbeddingAsym


impl <F> EmbeddingAsym<F> {

    pub(crate) fn new(source : Array2<F>, target : Array2<F>, distance : Distance<F>) -> Self {
        assert_eq!(source.dim().0, target.dim().0);
        assert_eq!(source.dim().1, target.dim().1);
        EmbeddingAsym{source, target, distance}
    }

    /// get representation of nodes as sources
    pub fn get_source(&self) -> &Array2<F> {
        &self.source
    }

    /// get representation of nodes as targets
    pub fn get_target(&self) -> &Array2<F> {
        &self.target
    }
 
} // end of impl block for EmbeddingAsym


impl<F>  EmbeddingT<F> for EmbeddingAsym<F> {

    fn is_symetric(&self) -> bool {
        return false;
    }

    /// get dimension of embedding. (row size of Array)
    fn get_dimension(&self) -> usize {
        self.source.dim().1
    }

    /// get distance from data1 to data2
    fn get_vec_distance(&self, data1 : &ArrayView1<F>, data2: &ArrayView1<F>) -> f64 {
        (self.distance)(data1, data2)
    }

    /// get distance FROM source node1 TO target node2 if embedding is asymetric, in symetric case there is no order) 
    fn get_node_distance(&self, node1 : usize, node2 : usize) -> f64 {
        (self.distance)(&self.source.row(node1), &self.target.row(node2))
    }
} // end impl EmbeddingT<F>