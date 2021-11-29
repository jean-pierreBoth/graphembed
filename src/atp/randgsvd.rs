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
// ndarray_linalg::Scalar provides Exp notation + Display + Debug + Serialize and sum on iterators

use anyhow::{Error, anyhow};

use num_traits::float::*;    // tp get FRAC_1_PI from FloatConst
use num_traits::cast::FromPrimitive;


use ndarray::{Array1, Array2};

use ndarray_linalg::{Scalar, Lapack};

// this module provides svdapproximation tools à la Hlako-Tropp
use annembed::tools::svdapprox::*;



// We fist implement a Range approximation with a precision criteria as 
// this one can be done with Sparse Matrix. Moreover it help determine the rank


/// We searh a generalized svd for the pair of matrix mat_1 (m,n) and mat_2 
/// i.e we search 2 orthogonal matrices V_1 and V_2 and one non singular patrix such that
/// Most often the Matrix Representation will be CSR and the precision approximation mode will
/// be used. But for Small graph we can consider the approximation with given target rank

pub struct GSvdApprox<'a, F: Scalar> {
    // matrix we want to approximate range of
    mat1 : &'a MatRepr<F>,
    //
    mat2 : &'a MatRepr<F>,
    //
    precision : RangeApproxMode,
}   // end of struct GsvdApprox



pub struct GSvdResult<F> {
    /// eigenvalues
    pub  v1 : Option<Array2<F>>,
    /// left eigenvectors. (m,r) matrix where r is rank asked for and m the number of data.
    pub  v2 : Option<Array2<F>>,
    /// first (diagonal matrix) eigenvalues
    pub  s1 : Option<Array1<F>>,
    /// first (diagonal matrix) eigenvalues
    pub  s2 : Option<Array1<F>>,
    /// common right term of mat1 and mat2 factorization
    pub commonx : Option<Array2<F>>
} // end of struct SvdResult<F> 



impl  <'a, F> GSvdApprox<'a, F>  
    where  F : Float + Lapack + Scalar  + ndarray::ScalarOperand + sprs::MulAcc {
    /// We impose the RangePrecision mode for now.
    pub fn new(mat1 : &'a MatRepr<F>, mat2 : &'a MatRepr<F>, precision : RangePrecision) -> Self {
        // TODO check for dimensions constraints, and type representation
        return GSvdApprox{mat1, mat2, precision : RangeApproxMode::EPSIL(precision)};
    } // end of new

    // We have to :
    //   - do a range approximation of the 2 matrices in problem definition
    //   - do a (full) gsvd of the 2 reduced matrices 

    /// 
    pub fn do_gsvd(&self) -> Result<GSvdResult<F>, anyhow::Error> {
        // We construct an approximation first for mat1 and then for mat2 and with the same precision 
        // criterion
        let r_approx1 = RangeApprox::new(self.mat1, self.precision);
        let  approx1_res = r_approx1.get_approximator();
        if approx1_res.is_none() {
            return Err(anyhow!("approximation of matrix 1 failed"));
        }
        let approx1_res = approx1_res.unwrap();
        let r_approx2 = RangeApprox::new(self.mat2, self.precision);
        let  approx2_res = r_approx2.get_approximator();
        if approx2_res.is_none() {
            return Err(anyhow!("approximation of matrix 2 failed"));
        }
        let approx2_res = approx2_res.unwrap();
        // We must not check for the ranks of approx1_res and approx2_res.
        // We want the 2 matrix to have the same weights but if we ran in precision mode we must
        // enforce that.
        // With Halko-Tropp (or Wei-Zhang and al.) conventions, we have mat1 = (m1,n), mat2 = (m2,n)
        // we get  approx1_res = (m1, l1)  and (m2, l2).
        // We must now construct reduced matrix approximating mat1 and mat2 i.e t(approx1_res)* mat1 
        // and t(approx2_res)* mat2
        let mut m = match self.mat1.get_data() {
            MatMode::FULL(mat) => { approx1_res.t().dot(mat)},
            MatMode::CSR(mat)  => { 
                                    log::trace!("direct_svd got csr matrix");
                                    small_transpose_dense_mult_csr(&approx1_res, mat)
                                },
        };
        let mut n = match self.mat2.get_data() {
            MatMode::FULL(mat) => { approx2_res.t().dot(mat)},
            MatMode::CSR(mat)  => { 
                                    log::trace!("direct_svd got csr matrix");
                                    small_transpose_dense_mult_csr(&approx2_res, mat)
                                },
        };
        // now we must do the standard generalized svd (with Lapack ggsvd3) for m and reduced_n
        // We are at step iv) of algo 2.4 of Wei and al.

        //
        Err(anyhow!("not yet implemented"))

    }  // end of do_svd

} // end of impl block for 