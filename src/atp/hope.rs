//! Asymetric Graph Embedding
//! Based on the paper:
//!     Asymetric Transitivity Preserving Graph Embedding 
//!     Ou, Cui Pei, Zhang, Zhu   in KDD 2016
//!  See  https://www.kdd.org/kdd2016/papers/files/rfp0184-ouA.pdf
//! 



use ndarray::{Array1, Array2};

use ndarray_linalg::{Scalar, Lapack};



use num_traits::float::*;    // tp get FRAC_1_PI from FloatConst
use num_traits::cast::FromPrimitive;

use sprs::prod;
use sprs::{CsMat};

use annembed::tools::svdapprox::{MatRepr};
/// Structure for asymetric embedding with approximate random generalized svd.
pub struct Hope<F> {
    /// the grap as a matrix
    mat : MatRepr<F>,
}



impl <F> Hope<F> {

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
    let mut v2 = Array1::<F>::from_elem(dims.0, F::zero());
    let mut iter = 0;
    let mut radius: F;
    let epsil = F::from_f64(1.0E-5).unwrap();
    loop {
        v2.fill(F::zero());
        let v2_slice = v2.as_slice_mut().unwrap();
        prod::mul_acc_mat_vec_csr(mat.view(), v1.as_slice().unwrap(), v2_slice);
        radius = Scalar::sqrt(v2.dot(&v2));
        v2 = v2 * F::one()/ radius;
        let w = &v1 - &v2;
        let delta = Scalar::sqrt(w.dot(&w));
        iter += 1;
        if iter >= 1000 || delta < epsil {
            log::info!(" estimated radius at iter {} {}", iter, radius.to_f64().unwrap());
            break;
        }
        v1.assign(&v2);
    }
    return radius.to_f64().unwrap();
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
        let w = &v1 - &v2;
        let delta = Scalar::sqrt(w.dot(&w));
        iter += 1;
        if iter >= 1000 || delta < epsil {
            log::info!(" estimated radius at iter {} {}", iter, radius.to_f64().unwrap());
            break;
        }
        v1 = v2;
    }
    //
    return radius.to_f64().unwrap();
} // end of estimate_spectral_radius_fullmat


//========================================================================================


mod tests {

    #[allow(unused)]
use super::*;

#[allow(unused)]
use sprs::{CsMat, TriMatBase};

#[allow(dead_code)]
fn log_init_test() {
    let _ = env_logger::builder().is_test(true).try_init();
}  


#[test]
fn test_spectral_radius_full() {
    log_init_test();
    log::info!("in test_spectral_radius_full");
    //
    let mat = ndarray::arr2(&[ [9., -1., 2.],
                                                                [-2., 8., 4.],
                                                                [1., 1., 8.]  ]);
    let radius = estimate_spectral_radius_fullmat(&mat);
    log::info!("estimate_spectral_radius_fullmat radius : {}", radius);
    //
    assert!((radius-10.).abs() < 1.0E-4);
} // end of test_spectral_radius_full


#[test]
fn test_spectral_radius_csr() {
    log_init_test();
    log::info!("in test_spectral_radius_csr"); 
    // compute radius from a full matrix
    let mat = ndarray::arr2(&[ [9., 0. , 2.],
        [0., 8., 4.],
        [1., 1., 0.]  ]);
        
    // check we get the same radius from the same matrix givn as a csr
    let mut rows = Vec::<usize>::with_capacity(6);
    let mut cols = Vec::<usize>::with_capacity(6);
    let mut values = Vec::<f64>::with_capacity(6);
    // row 0
    rows.push(0);
    cols.push(0);
    values.push(9.);

    rows.push(0);
    cols.push(2);
    values.push(2.);
    // row 1    
    rows.push(1);
    cols.push(1);
    values.push(8.);

    rows.push(1);
    cols.push(2);
    values.push(4.); 
    // row 2
    rows.push(2);
    cols.push(0);
    values.push(1.);

    rows.push(2);
    cols.push(1);
    values.push(1.);
    //
    let trimat = TriMatBase::<Vec<usize>, Vec<f64>>::from_triplets((3,3),rows, cols, values);
    let csr_mat : CsMat<f64> = trimat.to_csr();
    //
    let radius_from_full = estimate_spectral_radius_fullmat(&mat);
    log::info!("estimate_spectral_radius_fullmat radius : {}", radius_from_full);
    let radius_from_csmat = estimate_spectral_radius_csmat(&csr_mat);
    log::info!("estimate_spectral_radius_csmat radius : {}", radius_from_csmat);
    //
    assert!((radius_from_full-radius_from_csmat).abs() < 0.0001 * radius_from_full);


}  // enf of test_spectral_radius_csr



}  // end of mod test