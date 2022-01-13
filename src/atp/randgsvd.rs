//! implements a randomized generalized svd building upon
//! randomized svd.
//! We implement algorithm 2.4 from : 
//!     *Randomized General Singular Value Decomposition CAMC 2021*
//!     W. Wei, H. Zhang, X. Yang, X. Chen
//! We build upon the crate annembed which give us algo 2.3 of the same paper
//! (which corresponds to algo 4.2 of Halko-Tropp)



#![allow(unused)]
// num_traits::float::Float : Num + Copy + NumCast + PartialOrd + Neg<Output = Self>,  PartialOrd which is not in Scalar.
//     and nan() etc

// num_traits::Real : Num + Copy + NumCast + PartialOrd + Neg<Output = Self>
// as float but without nan() infinite() 

// ndarray::ScalarOperand provides array * F
// ndarray_linalg::Scalar provides Exp notation + Display + Debug + SerializFe and sum on iterators
use log::Level::Debug;
use log::{log_enabled};


use anyhow::{anyhow};

use num_traits::float::*;
//use num_traits::cast::FromPrimitive;


use ndarray::{s,Array1, Array2, ArrayView2, ArrayBase, ViewRepr, Dim, Ix1, Ix2};

use ndarray_linalg::{Scalar, Lapack};
use std::any::TypeId;

use lapacke::{dggsvd3, Layout};

use crate::embedding::{EmbeddingAsym};
use super::gsvd::{GSvd, GSvdOptParams, GSvdResult};


// this module provides svdapproximation tools à la Hlako-Tropp
use annembed::tools::svdapprox::*;



// We fist implement a Range approximation with a precision criteria as 
// this one can be done with Sparse Matrix. Moreover it help determine the rank
// Cannot use references in GSvdApprox beccause depending on Katz index/Rooted PageRank 
// it is not the same matrices that can be passed by reference!


#[cfg_attr(doc, katexit::katexit)]
/// We searh a generalized svd for the pair of matrix mat_1 (m,n) and mat_2 (p,n)
/// i.e we search 2 orthogonal matrices $V_1$ and $V_2$ , 2 diagonal matrices $\Sigma_{1}$ and $\Sigma_{1}$
/// and one non singular matrix X such that:
/// $$ V_{1}^{t} * mat1 * X = \Sigma_{1} \space and \space
/// 
///     V_{2}^{t} * mat2 * X = \Sigma_{2} $$
///  
/// The optional parameters can be used to modify (by multiplication or transposition) the 2 matrices mat1 and mat2.  
/// This avoids some matrix reallocations befote entering lapack.  
/// They are described in the GSvdOptParams documentation.
/// 
/// Most often the matrix representation will be CSR and the precision approximation mode will
/// be used. But for small graph we can consider the approximation with given target rank
/// 
pub struct GSvdApprox<F: Scalar> {
    /// first matrix we want to approximate range of
    mat1 : MatRepr<F>,
    /// second matrix
    mat2 : MatRepr<F>,
    /// optional parameters
    opt_params : Option<GSvdOptParams>,
    /// approximation mode
    precision : RangeApproxMode,
}   // end of struct GsvdApprox




impl  <F> GSvdApprox<F>  
    where  F : Float + Lapack + Scalar  + ndarray::ScalarOperand + sprs::MulAcc {
    /// We impose the RangePrecision mode for now.
    pub fn new(mat1 : MatRepr<F>, mat2 : MatRepr<F>, precision : RangePrecision, opt_params : Option<GSvdOptParams>) -> Self {
        // TODO check for dimensions constraints, and type representation

        return GSvdApprox{mat1, mat2, opt_params, precision : RangeApproxMode::EPSIL(precision)};
    } // end of new

    /// return optional paramertes if any
    pub fn get_parameters(&mut self,  alpha_1 : f64,  transpose_1 : bool,  alpha_2 : f64 , transpose_2 : bool) -> &Option<GSvdOptParams> {
        &self.opt_params
    } // end of set_parameters


    // We have to :
    //   - do a range approximation of the 2 matrices in problem definition
    //   - do a (full) gsvd of the 2 reduced matrices 
    //   - lapck rust interface requires we pass matrix as slices so they must be in row order!
    //     but for our application we must pass transposed version of Mg and Ml as we must compute inverse(Mg) * Ml
    //     with a = Mg and b = Ml. So it seems we cannot avoid copying when construction the GSvdApprox

    /// 
    pub fn do_approx_gsvd(&self) -> Result<GSvdResult<F>, anyhow::Error> {
        // We construct an approximation first for mat1 and then for mat2 and with the same precision 
        // criterion
        let r_approx1 = RangeApprox::new(&self.mat1, self.precision);
        let  approx1_res = r_approx1.get_approximator();
        if approx1_res.is_none() {
            return Err(anyhow!("approximation of matrix 1 failed"));
        }
        let approx1_res = approx1_res.unwrap();
        let r_approx2 = RangeApprox::new(&self.mat2, self.precision);
        let  approx2_res = r_approx2.get_approximator();
        if approx2_res.is_none() {
            return Err(anyhow!("approximation of matrix 2 failed"));
        }
        let approx2_res = approx2_res.unwrap();
        // We must not check for the ranks of approx1_res and approx2_res.
        // We want the 2 matrices to have the same weights but if we ran in precision mode we must
        // enforce that.
        // With Halko-Tropp (or Wei-Zhang and al.) conventions, we have mat1 = (m1,n), mat2 = (m2,n)
        // we get  approx1_res = (m1, l1)  and (m2, l2).
        // We must now construct reduced matrix approximating mat1 and mat2 i.e t(approx1_res)* mat1 
        // and t(approx2_res)* mat2 and get matrices (l1,n) and (l2,n)
        let mut a = match self.mat1.get_data() {
            MatMode::FULL(mat) => { approx1_res.t().dot(mat)},
            MatMode::CSR(mat)  => { 
                                    log::trace!("direct_svd got csr matrix");
                                    small_transpose_dense_mult_csr(&approx1_res, mat)
                                },
        };
        let mut b = match self.mat2.get_data() {
            MatMode::FULL(mat) => { approx2_res.t().dot(mat)},
            MatMode::CSR(mat)  => { 
                                    log::trace!("direct_svd got csr matrix");
                                    small_transpose_dense_mult_csr(&approx2_res, mat)
                                },
        };
        // now we must do the standard generalized svd (with Lapack ggsvd3) for m and reduced_n
        // We are at step iv) of algo 2.4 of Wei and al.
        let mut gsvd_pb = GSvd::new(&mut a,&mut b, None);
        let gsvd_res = gsvd_pb.do_gsvd(); 
        //
        if gsvd_res.is_err() {
            return Err(anyhow!("Gsvd failed")); 
        }

        gsvd_res
    }  // end of do_approx_gsvd

    // compute the approximated gsvd, builds the final embedding vectors.
    fn compute_embedding(&self) -> Result<EmbeddingAsym<F>, anyhow::Error>   {
        // do an approximate gsvd
        let gsvd_approx = self.do_approx_gsvd();
        if gsvd_approx.is_err() {
            return Err(anyhow!("Gsvd failed")); 
        }
        else {
            // must compute quotient of eigenvalues, check for null right eigenvalue.
            // possibly compute gsvd residual error.

        }
        return Err(anyhow!("compute_embedding not yet implemented")); 

    } // end of compute_embedding


    /// 
    pub fn compute_gsvd_residual(&self) -> f64 {
        panic!("not yet implemented");
        return -1.;
    }
} // end of impl block for GSvdApprox


//=========================================================================================================

mod tests {

#[allow(unused)]
use super::*;

use ndarray::{array};

use sprs::{CsMat, TriMat};

fn log_init_test() {
    let _ = env_logger::builder().is_test(true).try_init();
}  

// to convert a small matrix into a csr storage
fn smallmat_to_csr(a : &Array2<f64>) -> CsMat<f64> {
    let dim = a.dim();
    // fill the TriMat
    let mut trim = TriMat::new(a.dim());
    for (idx, val) in a.indexed_iter() {
        trim.add_triplet(idx.0, idx.1, *val);
    }
    // convert
    trim.to_csr()
}



fn test_gsvd_full_precision_1() {
    log_init_test();
    //
    let mat_a = array![ [1., 6., 11.],[2., 7., 12.] , [3., 8., 13.], [4., 9., 14.], [5., 10., 15.] ];
    let mat_b = array![ [8., 1., 6.],[3., 5., 7.] , [4., 9., 2.]];
    // 

} // end of test_gsv_full_1

// The smae test as test_gsvd_full_1 but with matrix described in csr mode, run in precision mode
fn test_gsvd_csr_precision_1() {
    log_init_test();
    //
    let mat_a = array![ [1., 6., 11.],[2., 7., 12.] , [3., 8., 13.], [4., 9., 14.], [5., 10., 15.] ];
    let mat_b = array![ [8., 1., 6.],[3., 5., 7.] , [4., 9., 2.]];
    // convert in csr mode !!
    let csr_a = smallmat_to_csr(&mat_a);
    let csr_b = smallmat_to_csr(&mat_b);
    let a = MatRepr::from_csrmat(csr_a);
    let b = MatRepr::from_csrmat(csr_b);

}  // end of test_gsvd_csr_precision_1




// we have full matrix we can test in rank mode
fn test_gsvd_full_rank_1() {
    log_init_test();
    //
    let mat_a = array![ [1., 6., 11.],[2., 7., 12.] , [3., 8., 13.], [4., 9., 14.], [5., 10., 15.] ];

    let mat_b = array![[8., 1., 6.],[3., 5., 7.] , [4., 9., 2.]];

} // end of test_gsvd_full_rank_1





} // end of mod tests    

