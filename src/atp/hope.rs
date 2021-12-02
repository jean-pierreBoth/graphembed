//! Asymetric Graph Embedding
//! Based on the paper:
//!     Asymetric Transitivity Preserving Graph Embedding 
//!     Ou, Cui Pei, Zhang, Zhu   in KDD 2016
//!  See  https://www.kdd.org/kdd2016/papers/files/rfp0184-ouA.pdf
//! 



use ndarray::{Dim, Array, Array1, Array2 , Ix1, Ix2};

use ndarray_linalg::{Scalar, Lapack};



use num_traits::float::*;    // tp get FRAC_1_PI from FloatConst
use num_traits::cast::FromPrimitive;

use sprs::prod;
use sprs::{CsMat};

struct Hope {

}



impl Hope {

    // return (I - β A, β A). We must check that beta is less than the spectral radius of adjacency matrix So the first term is inversible.
    // This ensure that the gsvd returned by lapack can be converted to the ATP paper. 
    fn make_katz_pair(&self, beta : f64) {

    } // end of make_katz_pair

 

    ///
    fn compute_embedding(&self) {
        // get spectral radius to decide on beta

        //  make katz pair 

        // transpose and formulate gsvd problem. 
        // We can now define a GSvdApprox structure

    }  // end of compute_embedding
}  // end of impl Hope



   // iterate in positive unit norm vector. mat is compressed row matrix and has all coefficients positive
fn estimate_spectral_radius_csmat<F>(mat : &CsMat<F>) -> f64 
        where F : Float + Scalar  + Lapack + ndarray::ScalarOperand + sprs::MulAcc  {
        //
    let dims = mat.shape();
    let init = F::from_f64(1./(dims.0 as f64).sqrt()).unwrap();
    let mut v1 = Array1::<F>::from_elem(dims.0, init);
    let mut v2 = v1.clone();
    let mut iter = 0;
    let mut radius: F;
    let epsil = F::from_f64(1.0E-5).unwrap();
    loop {

    }
    return 0.;
}   // end of estimate_spectral_radius_csmat



fn estimate_spectral_radius_fullmat<F>(mat : &Array2<F>) -> f64 
        where F : Float + Scalar  + Lapack + ndarray::ScalarOperand + sprs::MulAcc {
    let dims = mat.dim();
    let init = F::from_f64(1./(dims.0 as f64).sqrt()).unwrap();
    let mut v1 = Array1::<F>::from_elem(dims.0, init);
    let mut v2: Array1::<F>;
    let mut iter = 0;
    let epsil = F::from_f64(1.0E-5).unwrap();
    let mut radius : F;
    loop {
        v2 = mat.dot(&v1);
        radius = Scalar::sqrt(v2.dot(&v2));
        v2 = v2 * F::one()/ radius;
        let w = v1 - &v2;
        let delta = Scalar::sqrt(w.dot(&w));
        iter += 1;
        if iter >= 100 || delta < epsil {
            log::debug!(" estimated radius at iter {} {}", iter, radius.to_f64().unwrap());
            break;
        }
        v1 = v2;
    }
    //
    return radius.to_f64().unwrap();
} // end of estimate_spectral_radius_fullmat