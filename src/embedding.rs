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



use ndarray::{Array2, Array1};


pub trait EmbeddingT<F> {
    /// returns true if embedding is symetric
    fn is_symetric(&self) -> bool;
    /// the trait provides a function (distance or similarity) between embedded items
    fn get_similarity(v1 : &Array1<F>, v2: &Array1<F>) -> f64;
    ///
    fn get_dimension() -> usize;
}
/// symetric embedding
pub struct Embedding<F> {
    /// array (n,d) with n number of data, d dimension of embedding
    data: Array2<F>,
    /// distance
    distance : fn(&Array1<F>, &Array1<F>) -> f64,
} // end of Embedding



impl<F> Embedding<F> {

    pub(crate) fn new(arr : Array2<F>, distance : fn(&Array1<F>, &Array1<F>) -> f64) -> Self {
        Embedding{data : arr, distance : distance}
    }

    /// get dimension of embedding. (row size of Array)
    pub fn get_dimension(&self) -> usize {
        self.data.dim().1
    }
}  // end of impl Embedding



//============================================


/// Asymetric embedding representation
pub struct EmbeddingAsym<F> {
    /// source node representation
    source : Array2<F>,
    /// target node representation
    target : Array2<F>,
} // end of struct EmbeddingAsym


impl <F> EmbeddingAsym<F> {

    pub(crate) fn new(source : Array2<F>, target : Array2<F>) -> Self {
        assert_eq!(source.dim().0, target.dim().0);
        assert_eq!(source.dim().1, target.dim().1);
        EmbeddingAsym{source, target}
    }

    pub fn get_source(&self) -> &Array2<F> {
        &self.source
    }

    pub fn get_target(&self) -> &Array2<F> {
        &self.target
    }
 
    /// get dimension of embedding. (row size of Array)
    pub fn get_dimension(&self) -> usize {
        self.source.dim().1
    }
} // end of impl block for EmbeddingAsym