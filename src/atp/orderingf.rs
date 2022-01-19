//! Some utilities to order the abstract type for f64 or f32 
//! i.e trait satisfying F : Float + Scalar  + Lapack + ndarray::ScalarOperand + sprs::MulAcc + for<'r> std::ops::MulAssign<&'r F> + Default


use num_traits::float::*; 
use std::cmp;

/// indexed value to keep track of position after sorting
#[derive(Copy, Clone)]
pub struct IndexedValue<F>(pub usize, pub F);


/// makes an ordering on Float by putting Nan at end of sort.
/// rust sorts in Increasing order so we reverse Greater and Less
pub(crate) fn decreasing_sort_nans_first<F : Float>(a: &IndexedValue<F>, b: &IndexedValue<F>) -> cmp::Ordering {
    match (a, b) {
        (x, y) if x.1.is_nan() && y.1.is_nan() => cmp::Ordering::Equal,
        (x, _) if x.1.is_nan() => cmp::Ordering::Less,
        (_, y) if y.1.is_nan() => cmp::Ordering::Greater,
        (_, _) => b.partial_cmp(a).unwrap()
    }
}  // end of decreasing_sort_nans_first


impl  <F:PartialOrd+PartialEq> PartialEq for IndexedValue<F> {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}

impl  <F:PartialOrd+PartialEq> Eq for IndexedValue<F> {}



/// implement an order on IndexedValue.
/// To be used with Vec<IndexedValue<F>>::unstable_sort
impl <F:PartialOrd+PartialEq>  PartialOrd for IndexedValue<F> {
    fn partial_cmp(&self , other: &Self) -> Option<cmp::Ordering> {
        self.1.partial_cmp(&other.1)
    }
} // end of PartialOrd
