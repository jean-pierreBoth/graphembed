//! Asymetric Graph Embedding
//! Based on the paper:
//!     Asymetric Transitivity Preserving Graph Embedding 
//!     Ou, Cui Pei, Zhang, Zhu   in KDD 2016
//!  See  https://www.kdd.org/kdd2016/papers/files/rfp0184-ouA.pdf
//! 


struct Hope {

}



impl Hope {

    // return (I - β A, β A). We must check that beta is less than the spectral radius of adjacency matrix So the first term is inversible.
    // This ensure that the gsvd returned by lapack can be converted to the ATP paper. 
    fn make_katz_pair(&self, beta : f64) {

    } // end of make_katz_pair

    // iterate in positive unit norm vector
    fn estimate_spectral_radius(&self) -> f64 {

    }   // end of estimate_spectral_radius


    ///
    fn compute_embedding(&self) {
        // get spectral radius to decide on beta

        //  make katz pair 

        // transpose and formulate gsvd problem. 
        // We can now define a GSvdApprox structure

    }  // end of compute_embedding
}  // end of impl Hope
