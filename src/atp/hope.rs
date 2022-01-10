//! Asymetric Graph Embedding
//! Based on the paper:
//!     Asymetric Transitivity Preserving Graph Embedding 
//!     Ou, Cui Pei, Zhang, Zhu   in KDD 2016
//!  See  [atp](https://www.kdd.org/kdd2016/papers/files/rfp0184-ouA.pdf)
//! 



use ndarray::{Array1, Array2};

use ndarray_linalg::{Scalar, Lapack};



use num_traits::float::*;    // tp get FRAC_1_PI from FloatConst
// use num_traits::cast::FromPrimitive;

use sprs::prod;
use sprs::{CsMat, TriMatBase};

use annembed::tools::svdapprox::{MatRepr, MatMode, RangePrecision};

use super::randgsvd::{GSvdApprox};
use super::gsvd::{GSvdOptParams};
/// Structure for graph asymetric embedding with approximate random generalized svd to get an estimate of rank necessary
/// to get a required precision in the SVD. 
/// The structure stores the adjacency matrix in a full (ndarray) or compressed row storage format (using crate sprs).
pub struct Hope<F> {
    /// the graph as a matrix
    mat : MatRepr<F>,
}



impl <F> Hope<F>  where
    F: Float + Scalar  + Lapack + ndarray::ScalarOperand + sprs::MulAcc {

    /// instantiate a Hope problem with the adjacency matrix
    pub fn from_ndarray(mat : Array2<F>) -> Self {
        let mat = MatRepr::from_array2(mat);
        Hope::<F> {mat}
    }


    // Noting A the adjacency matrix we constitute the couple (M_g, M_l ) = (I - β A, β A). 
    // We must check that beta is less than the spectral radius of adjacency matrix so that M_g is inversible.
    // In fact we go to the Gsvd with the pair (transpose(β A), transpose(I - β A))

    /// - factor helps defining the extent to which the neighbourhood of a node is taken into account when using the katz index matrix.
    ///     factor must be between 0. and 1.
    #[allow(unused)]
    fn make_katz_problem(&self, factor : f64) {
        // enforce rule on factor

        //
        let radius = match self.mat.get_data() {
            MatMode::FULL(mat) =>  { estimate_spectral_radius_fullmat(&mat) },
            MatMode::CSR(csmat) =>  { estimate_spectral_radius_csmat(&csmat)},
        };        
        //  defining beta ensures that the matrix (Mg) in Hope paper is inversible.
        let beta = factor / radius;
        // now we can define a GSvdApprox problem
        let epsil = 0.1;
        let rank_increment = 20;
        let max_rank = 300;
        let rangeprecision = RangePrecision::new(epsil, rank_increment, max_rank);
        // We must now define mat1 and mat2 (or A and B  in Wei-Zhang paper)
        // mat1 is easy: it is beta * transpose(self.mat), so we define mat1 by &self.mat and adjust optional parameters
        // accordingly.
        // For mat2 it is I - beta * &self.mat, we need to reallocate a matrix
        let mat1 = &self.mat;
        // TODO compute mat2
        let opt_params = GSvdOptParams::new(beta, true, 1., true);
        let mat2 = compute_1_minus_beta_mat(&self.mat, beta);
        let gsvdapprox = GSvdApprox::new(mat1, &mat2 , rangeprecision, Some(opt_params));
        // now we can solve svd problem
        let _gsvd_res = gsvdapprox.do_approx_gsvd();
    } // end of make_katz_pair


    /// 
    #[allow(unused)]
    fn make_rooted_pagerank_problem(&self) {
        panic!("make_rooted_pagerank_problem: not yet implemented");
    } // end of make_rooted_pagerank_problem

    /// computes the embedding.
    /// - factor helps defining the extent to which the neighbourhood of a node is taken into account when using the katz index matrix.
    ///     factor must be between 0. and 1.
    /// 
    pub fn compute_embedding(&self) {
    

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


// useful for Katz Index
fn compute_1_minus_beta_mat<F>(mat : &MatRepr<F>, beta : f64) -> MatRepr<F> 
        where  F : Float + Scalar  + Lapack + ndarray::ScalarOperand + sprs::MulAcc {
            //
    match mat.get_data() {
        MatMode::FULL(mat) => {
            let new_mat = mat * F::from_f64(beta).unwrap();
            return MatRepr::from_array2(new_mat);
        },
        // TODO can we do better ?
        MatMode::CSR(mat)  => { 
            log::trace!("compute_1_minus_beta_mat");
            assert_eq!(mat.rows(), mat.cols());
            let n = mat.rows();
            let nnz = mat.nnz();
            // get an iter on triplets, construct a new trimatb
            let mut rows = Vec::<usize>::with_capacity(nnz+ n);
            let mut cols = Vec::<usize>::with_capacity(nnz+n);
            let mut values = Vec::<F>::with_capacity(nnz+n);
            let mut already_diag = Array1::<u8>::zeros(n);
            let mut iter = mat.iter();
            let beta_f = F::from_f64(beta).unwrap();
            while let Some((val, (row, col))) = iter.next() {
                if row != col {
                    rows.push(row);
                    cols.push(col);
                    values.push(- beta_f * *val);

                } else {
                    values.push(F::one() - beta_f * *val);
                    already_diag[row] = 1;
                }
            };
            // fill values in diag not already initialized
            for i in 0..n {
                if already_diag[i] == 0 {
                    rows.push(i);
                    cols.push(i);
                    values.push(F::one()); 
                }                   
            }
            let trimat = TriMatBase::<Vec<usize>, Vec<F>>::from_triplets((4,5),rows, cols, values);
            let csr_mat : CsMat<F> = trimat.to_csr();
            return MatRepr::from_csrmat(csr_mat);
        },
    }  // end of match
} // end of compute_1_minus_beta_mat


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

// example from https://en.wikipedia.org/wiki/Spectral_radius
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