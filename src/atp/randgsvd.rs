//! implements a randomized generalized svd building upon
//! randomized svd

// num_traits::float::Float : Num + Copy + NumCast + PartialOrd + Neg<Output = Self>,  PartialOrd which is not in Scalar.
//     and nan() etc

// num_traits::Real : Num + Copy + NumCast + PartialOrd + Neg<Output = Self>
// as float but without nan() infinite() 

// ndarray::ScalarOperand provides array * F
// ndarray_linalg::Scalar provides Exp notation + Display + Debug + Serialize and sum on iterators


use ndarray_linalg::{Scalar, Lapack};

use annembed::tools::svdapprox::*;



// We fist implement a Range approximation with a precision criteria as 
// this one can be done with Sparse Matrix. Moreover it help determine the rank


/// We searh a generalized svd for the pair of matrix mat_1 and mat_2
/// i.e we search 2 orthogonal matrices V_1 and V_2 and one non singular patrix such that
/// ```math
/// V\_{1}^{t} * M_{1} * X = \Sigma_{1}
/// V\_{2}^{t} * M_{2} * X = \Sigma_{2}
/// where \Sigma_{1} and \Sigma_{2} are two diagonal matrices.
/// ```
///  
struct GsvdApprox <'a, F: Scalar> {
        /// matrix we want to approximate range of.
        mat_1 : &'a MatRepr<F>,
        //
        mat_2 : &'a MatRepr<F>
}   // end of struct GsvdApprox