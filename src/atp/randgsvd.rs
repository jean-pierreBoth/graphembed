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
use log::Level::Debug;
use log::{debug, log_enabled};


use anyhow::{Error, anyhow};

use num_traits::float::*;    // tp get FRAC_1_PI from FloatConst
use num_traits::cast::FromPrimitive;


use ndarray::{s,Array1, Array2, ArrayView2, ArrayBase, ViewRepr, Dim, Ix1, Ix2};

use ndarray_linalg::{Scalar, Lapack};
use std::any::TypeId;

use lapacke::{dggsvd3, Layout};

// this module provides svdapproximation tools à la Hlako-Tropp
use annembed::tools::svdapprox::*;



// We fist implement a Range approximation with a precision criteria as 
// this one can be done with Sparse Matrix. Moreover it help determine the rank


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
pub struct GSvdApprox<'a, F: Scalar> {
    /// first matrix we want to approximate range of
    mat1 : &'a MatRepr<F>,
    /// second matrix
    mat2 : &'a MatRepr<F>,
    /// optional parameters
    opt_params : Option<GSvdOptParams>,
    /// approximation mode
    precision : RangeApproxMode,
}   // end of struct GsvdApprox


#[derive(Copy, Clone, Debug)]
/// This structure describes optional parameters used to specify the Gsvd approximation to do by GSvdApprox
/// It can be useful to keep the two matrices mat1 and mat2 stored in GSvdApprox in one order but to solve the problem for their transpose
/// (as is the case in the Hope algorithm).  
/// In this case the transpose flags are used to send to lapack the matrices with a transpose flag.
/// For the multplication factor (also useful in the Hope algorithm they are applied in a later stage of the algorithm) 
pub struct GSvdOptParams {
    /// multiplication factor to use for mat1. default to 1.
    alpha_1 : f64, 
    /// transposition to apply to mat1. default to no
    transpose_1 : bool,
    /// multiplication factor to use for mat2. default to 1.
    alpha_2 : f64, 
    /// transposition to apply to mat2? default to no
    transpose_2 : bool,    
}  // end of struct GSvdOptParams


impl GSvdOptParams {
    pub fn new(alpha_1 : f64,  transpose_1 : bool,  alpha_2 : f64 , transpose_2 : bool) -> Self {
        GSvdOptParams {alpha_1, transpose_1, alpha_2, transpose_2}   
    } // end of new GSvdOptParams

    pub fn get_alpha_1(&self) -> f64 { self. alpha_1}

    pub fn get_alpha_2(&self) -> f64 { self. alpha_2}

    pub fn get_transpose_1(&self) -> bool { self.transpose_1}

    pub fn get_transpose_2(&self) -> bool { self.transpose_2}

} // end of impl GSvdOptParams


#[cfg_attr(doc, katexit::katexit)]
/// For a problem described in GSvdApprox by the pair of matrix mat_1 (m,n) and mat_2 (p,n)
/// we get:  
/// 
///  - 2 orthogonal matrices  $V_{1}$  and  $V_{2}$
///       
///  - 2 diagonal matrices $\Sigma_{1}$ and $\Sigma_{1}$  
/// 
///  - one non singular matrix X such that:
/// $$ V_{1}^{t} * mat1 * X = \Sigma_{1} \space and \space
///    V_{2}^{t} * mat2 * X = \Sigma_{2} $$
/// 
pub struct GSvdResult<F: Float + Scalar> {
    /// left eigenvectors for first matrix. U
    pub(crate)  v1 : Option<Array2<F>>,
    /// left eigenvectors. (m,r) matrix where r is rank asked for and m the number of data.
    pub(crate)  v2 : Option<Array2<F>>,
    /// first (diagonal matrix) eigenvalues
    pub(crate)  s1 : Option<Array1<F>>,
    /// second (diagonal matrix) eigenvalues
    pub(crate)  s2 : Option<Array1<F>>,
    /// common right term of mat1 and mat2 factorization
    pub(crate) commonx : Option<Array2<F>>
} // end of struct SvdResult<F> 


impl <F> GSvdResult<F>  where  F : Float + Lapack + Scalar + ndarray::ScalarOperand + sprs::MulAcc  {

    pub(crate) fn new() -> Self {
        GSvdResult{v1 :None, v2 : None, s1 : None, s2 : None, commonx :None}
    }

    // reconstruct result from the out parameters of lapack. For us u and v are always asked for
    // (m,n) is dimension of A. p is number of rows of B. k and l oare lapack output  
    pub(crate) fn init_from_lapack(&mut self, m : i64, n : i64, p : i64, u : Array2<F>, v : Array2<F>, k : i64 ,l : i64 , 
                alpha : Array1<F>, beta : Array1<F>, permuta : Array1<i32>) {
        self.v1 = Some(u);
        self.v2 = Some(v);
        // now we must decode depending upon k and l values, we use the lapack doc at :
        // http://www.netlib.org/lapack/explore-html/d1/d7e/group__double_g_esing_gab6c743f531c1b87922eb811cbc3ef645.html
        //
        log::debug!("m : {}, n : {}, k : {}, l : {} ", m, n, k, l);
        assert!(m >= 0);
        assert!(l >= 0);
        assert!(k >= 0);
        //
        let s1_v : ArrayBase<ViewRepr<&F>, Dim<[usize;1]>>;
        let s2_v : ArrayBase<ViewRepr<&F>, Dim<[usize;1]>>;
        // on 0..k  alpha = 1. beta = 0.
        if m-k-l >= 0 {
            log::debug!("m-k-l >= 0");
            // s1 is alpha[k .. k+l-1] and   s2 is beta[k .. k+l-1], 
            assert!(l > 0);
            assert!(k >= 0);
            s1_v = alpha.slice(s![k as usize ..(k+l) as usize]);
            s2_v = beta.slice(s![k as usize ..(k+l) as usize]);
        }
        else {
            log::debug!("m-k-l < 0");
            // s1 is alpha[k..m]  and s2 is beta[k..m], alpha[m..k+l] == 0 and beta[m..k+l] == 1 and beyond k+l  alpha = beta == 0
            assert!(k >= 0);           
            assert!(m >= k);
            s1_v = alpha.slice(s![k as usize..(m as usize)]);
            s2_v = beta.slice(s![k as usize..(m as usize)]);
        }
        // a dump if log Debug enabled we dump C and S
        if log_enabled!(Debug) {
            for i in 0..s1_v.len() {
                log::debug!(" i {}, S[i] {},  C[i] {}", i, s1_v[i], s2_v[i]);
            }
        }
        // some checks
        let check : Vec<F> = s1_v.iter().zip(s2_v.iter()).map(| x |  *x.0 * *x.0 + *x.1 * *x.1).collect();
        for v in check {
            let epsil = (1. - v.to_f64().unwrap()).abs();
            log::debug!(" epsil = {}", epsil);
            assert!(epsil < 1.0E-5 );
        }
        // we clone
        let s1 = s1_v.to_owned();
        let s2 = s2_v.to_owned();
        // TODO monotonicity check.  before clone ?? It seems alpha is sorted.
        let mut alpha_sorted = alpha.clone();
        let s = m.min(k+l) as usize;
        let k_1 = k as usize;
 /*        for i in k_1..s {
            log::debug!("i {} , permuta[i] {}", i , permuta[i]);
            alpha_sorted.swap(i, permuta[i] as usize);
        }  */
        for i in k_1+1..s {
            if alpha[i] > alpha[i-1] {
                log::error!("alpha non decreasing at i : {}  {}  {}", i, alpha[i], alpha[i-1]);
                panic!("non sorted alpha");
            }
        }
        // possibly commonx (or Q in Lapack docs) but here we do not keep it
    }  // end of GSvdResult::init_from_lapack

    // debug utility for tests!
    fn dump_u(&self) {
        if self.v1.is_some() {
            let u = self.v1.as_ref().unwrap();
            println!("dumping U");
            dump::<F>(&u.view());
        }
    }  // end of dump_u

    fn check_uv_orthogonal(&self) {
        if self.v1.is_some() {
            let u = self.v1.as_ref().unwrap();
            let id : Array2<F> = u.dot(&u.t());       
            println!("\n\n dumping u*tu");
            dump(&id.view());
        }
        if self.v2.is_some() {
            let v = self.v2.as_ref().unwrap();
            println!("dumping v");
            dump::<F>(&v.view());
            let id : Array2<F> = v.dot(&v.t());       
            println!("\n\n dumping v*tv");
            dump(&id.view());
        }
    }  // end of check_u_orthogonal

} // end of impl block for GSvdResult


fn dump<F>(a : &ArrayView2<F>) where F : Float + Lapack + Scalar {
    for i in 0..a.dim().0 {
        println!();
        for j in 0..a.dim().1 {
            print!("{:.3e} ", a[[i,j]]);
        }
    }
} // end of dump

impl  <'a, F> GSvdApprox<'a, F>  
    where  F : Float + Lapack + Scalar  + ndarray::ScalarOperand + sprs::MulAcc {
    /// We impose the RangePrecision mode for now.
    pub fn new(mat1 : &'a MatRepr<F>, mat2 : &'a MatRepr<F>, precision : RangePrecision, opt_params : Option<GSvdOptParams>) -> Self {
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
        // See rust doc https://docs.rs/lapacke/latest/lapacke/fn.dggsvd3.html and
        // fortran https://www.netlib.org/lapack/lug/node36.html#1815 but new function is (s|d)ggsvd3
        //
        // Lapack definition of GSVD is in the following link:
        // http://www.netlib.org/lapack/explore-html/d1/d7e/group__double_g_esing_gab6c743f531c1b87922eb811cbc3ef645.html
        //
        //  Lapack GSVD(A,B) for A=(m,n) and B=(p,n) 
        //  gives U**T*A*Q = D1*( 0 R ),    V**T*B*Q = D2*( 0 R )   with  U , V and Q orthogonals
        //
        let (a_nbrow, a_nbcol) = a.dim();
        let jobu = b'U';
        let jobv = b'V';
        let jobq = b'N';        // Q is large we do not need it, we do not compute it
        assert_eq!(a_nbcol, b.dim().1); // check m and n have the same number of columns.
        let mut k : i32 = 0;
        let mut l : i32 = 0;
        // for lda  see lapacke interface  : http://www.netlib.org/lapack/lapacke.html#_array_arguments
        // Caution our matrix are C (row) ordered so lda is nbcol. but we want to send the transpose (!) so lda is a_nbrow
        let lda : i32 = a_nbrow as i32;
        let b_dim = b.dim();
        // caution our matrix are C (row) ordered so lda is nbcol. but we want to send the transpose (!) so lda is a_nbrow
        let ldb : i32 = b_dim.0 as i32;
        let ires: i32;
        let ldu = lda;  // TODO check that : as we compute U , ldu must be greater than nb rows of A
        let ldv = ldb;
        //
        let ldq : i32 = a_nbcol as i32;  // as we do not ask for Q but test test_lapack_array showed we cannot set to 1!
        let mut iwork = Array1::<i32>::zeros(a_nbcol);
        let u : Array2::<F>;
        let v : Array2::<F>;
        let alpha : Array1::<F>;
        let beta : Array1::<F>;
        let mut gsvdres = GSvdResult::<F>::new();
        //
        if TypeId::of::<F>() == TypeId::of::<f32>() {
            let mut alpha_f32 = Vec::<f32>::with_capacity(a_nbcol);
            let mut beta_f32 = Vec::<f32>::with_capacity(a_nbcol);
            let mut u_f32= Array2::<f32>::zeros((a_nbrow, a_nbrow));
            let mut v_f32= Array2::<f32>::zeros((b_dim.0, b_dim.0));
            let mut q_f32 = Vec::<f32>::new();
            ires = unsafe {
                // we must cast a and b to f32 slices!! unsafe but we know our types with TypeId
                let mut af32 = std::slice::from_raw_parts_mut(a.as_slice_mut().unwrap().as_ptr() as * mut f32 , a.len());
                let mut bf32 = std::slice::from_raw_parts_mut(b.as_slice_mut().unwrap().as_ptr() as * mut f32 , b.len());
                let ires = lapacke::sggsvd3(Layout::RowMajor, jobu, jobv, jobq, 
                        //nb row of m , nb columns , nb row of n
                        a_nbrow.try_into().unwrap(), a_nbcol.try_into().unwrap(), b.dim().0.try_into().unwrap(),
                        &mut k, &mut l,
                        &mut af32, lda,
                        &mut bf32, ldb,
                        alpha_f32.as_mut_slice(),beta_f32.as_mut_slice(),
                        u_f32.as_slice_mut().unwrap(), ldu,
                        v_f32.as_slice_mut().unwrap(), ldv,
                        q_f32.as_mut_slice(), ldq,
                        iwork.as_slice_mut().unwrap());
                if ires == 0 {
                    // but now we must  transform u,v, alpha and beta from f32 to F
                    u = ndarray::ArrayView::<F, Ix2>::from_shape_ptr(u_f32.dim(), u_f32.as_ptr() as *const F).into_owned();
                    v = ndarray::ArrayView::<F, Ix2>::from_shape_ptr(v_f32.dim(), v_f32.as_ptr() as *const F).into_owned();
                    alpha = ndarray::ArrayView::<F, Ix1>::from_shape_ptr((alpha_f32.len()),alpha_f32.as_ptr() as *const F).into_owned();
                    beta = ndarray::ArrayView::<F, Ix1>::from_shape_ptr((beta_f32.len()),beta_f32.as_ptr() as *const F).into_owned();
                    // convert usize to i64 as matrix sizes surely permits that
                    gsvdres.init_from_lapack(a_nbrow.try_into().unwrap(), a_nbcol.try_into().unwrap() , b_dim.0.try_into().unwrap(), 
                                u, v, k as i64, l as i64 , alpha , beta, iwork);
                }
                else if ires == 1 {
                    return Err(anyhow!("lapack failed to converge"));
                }
                else if ires < 0 {
                    return Err(anyhow!("argument {} had an illegal value", -ires));
                }
                //
                ires
            }; // end of unsafe block
        }  // end case f32
        else if TypeId::of::<F>() == TypeId::of::<f64>() {
            let mut alpha_f64 = Vec::<f64>::with_capacity(a_nbcol);
            let mut beta_f64 = Vec::<f64>::with_capacity(a_nbcol);
            let mut u_f64= Array2::<f64>::zeros((a_nbrow, a_nbrow));
            let mut v_f64= Array2::<f64>::zeros((b_dim.0, b_dim.0));
            let mut q_f64 = Vec::<f64>::new(); 
            ires = unsafe {
                let mut af64 = std::slice::from_raw_parts_mut(a.as_slice_mut().unwrap().as_ptr() as * mut f64 , a.len());
                let mut bf64 = std::slice::from_raw_parts_mut(b.as_slice_mut().unwrap().as_ptr() as * mut f64 , b.len()); 
                let ires = lapacke::dggsvd3(Layout::RowMajor, jobu, jobv, jobq, 
                    //nb row of m , nb columns , nb row of n
                    a_nbrow.try_into().unwrap(), a_nbcol.try_into().unwrap(), b.dim().0.try_into().unwrap(),
                    &mut k, &mut l,
                    &mut af64, lda,
                    &mut bf64, ldb,
                    alpha_f64.as_mut_slice(),beta_f64.as_mut_slice(),
                    u_f64.as_slice_mut().unwrap(), ldu,
                    v_f64.as_slice_mut().unwrap(), ldv,
                    q_f64.as_mut_slice(), ldq,
                    iwork.as_slice_mut().unwrap());
                // but now we must transform u,v, alpha and beta from f64 to F
                if ires == 0 {
                    u = ndarray::ArrayView::<F, Ix2>::from_shape_ptr(u_f64.dim(), u_f64.as_ptr() as *const F).into_owned();
                    v = ndarray::ArrayView::<F, Ix2>::from_shape_ptr(v_f64.dim(), v_f64.as_ptr() as *const F).into_owned();
                    alpha = ndarray::ArrayView::<F, Ix1>::from_shape_ptr((alpha_f64.len()),alpha_f64.as_ptr() as *const F).into_owned();
                    beta = ndarray::ArrayView::<F, Ix1>::from_shape_ptr((beta_f64.len()),beta_f64.as_ptr() as *const F).into_owned();
                    gsvdres.init_from_lapack(a_nbrow.try_into().unwrap(), a_nbcol.try_into().unwrap() , b_dim.0.try_into().unwrap(), 
                            u, v, k as i64, l as i64 , alpha , beta, iwork);
                }
                else if ires == 1 {
                    return Err(anyhow!("lapack failed to converge"));
                }
                else if ires < 0 {
                    return Err(anyhow!("argument {} had an illegal value", -ires));
                }                
                ires
            }  // end unsafe         
        }  // end case f64
        else {
            log::error!("do_approx_gsvd only implemented for f32 and f64");
            panic!();
        }
        Ok(gsvdres)
    }  // end of do_approx_gsvd

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


// with more rows than columns.run in precision mode


fn small_lapack_gsvd(a: &mut Array2<f64>, b : &mut Array2<f64>) -> GSvdResult::<f64> {
    //
    let (a_nbrow, a_nbcol) = a.dim();
    let jobu = 'U' as u8;   // we compute U
    let jobv = b'V';   // we compute V
    let jobq = 'N' as u8;   // Q is large we do not need it, we do not compute it
    assert_eq!(a_nbcol, b.dim().1); // check m and n have the same number of columns.
    let mut k : i32 = 0;
    let mut l : i32 = 0;
    let lda : i32 = a_nbcol as i32;  // our matrix are row ordered and see https://www.netlib.org/lapack/lapacke.html
    let b_dim = b.dim();
    let ldb : i32 = b_dim.1 as i32;     // our matrix are row ordered!
    let mut alpha_f64 = Array1::<f64>::zeros(a_nbcol);
    let mut beta_f64 = Array1::<f64>::zeros(a_nbcol);
    let mut u_f64= Array2::<f64>::zeros((a_nbrow, a_nbrow));
    let mut v_f64= Array2::<f64>::zeros((b_dim.0, b_dim.0));
    let mut q_f64 = Vec::<f64>::new(); 
    let ldu = a_nbrow as i32;  // as we compute U , ldu must be greater than nb rows of A
    let ldv = b_dim.1 as i32;
    // The following deviates from doc http://www.netlib.org/lapack/explore-html/d1/d7e/group__double_g_esing_gab6c743f531c1b87922eb811cbc3ef645.html
    let ldq = a_nbcol as i32;    // we do not ask for Q but ldq must be >= a_nbcol (error msg from LAPACKE_dggsvd3_work)
    let mut iwork = Array1::<i32>::zeros(a_nbcol);
    // lda parameter 11, ldv parameter 19  in dggsvd3 and 
    let ires = unsafe {
        let mut a_slice = std::slice::from_raw_parts_mut(a.as_slice_mut().unwrap().as_ptr() as * mut f64 , a.len());
        let mut b_slice = std::slice::from_raw_parts_mut(b.as_slice_mut().unwrap().as_ptr() as * mut f64 , b.len()); 
        dggsvd3(Layout::RowMajor, jobu, jobv, jobq, 
                //nb row of m , nb columns , nb row of n
                a_nbrow.try_into().unwrap(), a_nbcol.try_into().unwrap(), b.dim().0.try_into().unwrap(),
                &mut k, &mut l,
                a_slice, lda, b_slice, ldb,
                alpha_f64.as_slice_mut().unwrap(),beta_f64.as_slice_mut().unwrap(),
                u_f64.as_slice_mut().unwrap(), ldu,
                v_f64.as_slice_mut().unwrap(), ldv,
                q_f64.as_mut_slice(), ldq,
                iwork.as_slice_mut().unwrap()
        )
    };
    // 
    if ires != 0 {
        println!("ggsvd3 returned {}", ires);
        log::error!("dggsvd3 returned {}", ires);
        assert!(1==0);
    }
    log::debug!("dggsvd3 passed");
    // allocate result
    let mut gsvdres = GSvdResult::<f64>::new();
    gsvdres.init_from_lapack(a_nbrow.try_into().unwrap(), a_nbcol.try_into().unwrap(), b_dim.0.try_into().unwrap(), 
            u_f64, v_f64, k.into(), l.into(), alpha_f64, beta_f64, iwork);
    //
    gsvdres
}   // end of small_lapack_gsvd






#[test]
// a test to check rust lapack interface more rows than columns
// small example from https://fr.mathworks.com/help/matlab/ref/gsvd.html
fn test_lapack_gsvd_array_1() {
    log_init_test();
    //
    let mut a = array![ [1., 6., 11.], [2., 7., 12.] , [3., 8., 13.], [4., 9., 14.], [5., 10., 15.] ];
    let mut b = array![ [8., 1., 6.],[3., 5., 7.] , [4., 9., 2.]];
    let gsvdres = small_lapack_gsvd(&mut a, &mut b);
    // dump results
    gsvdres.dump_u();
    gsvdres.check_uv_orthogonal();
} // end of test_lapack_gsvd_array




// taken from https://rdrr.io/cran/geigen/man/gsvd.html
#[test]
fn test_lapack_gsvd_array_2() {
    log_init_test();
    //
    let mut a = array![ [ 1. , 2. , 3. , 3.,  2. , 1.] , [ 4. , 5. , 6. , 7. , 8., 8.]   ];
    let mut b = array![ [1., 2., 3., 4., 5., 6.] , 
                                                [ 7. , 8., 9., 10., 11., 12.] , 
                                                [ 13. , 14., 15., 16., 17., 18.]   ];
    let gsvdres = small_lapack_gsvd(&mut a, &mut b);
    // dump results
    gsvdres.dump_u();
    gsvdres.check_uv_orthogonal();
} // end of test_lapack_gsvd_array_2


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

