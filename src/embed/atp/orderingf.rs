//! Some utilities to order the abstract type for f64 or f32 
//! i.e trait satisfying F : Float + Scalar  + Lapack + ndarray::ScalarOperand + sprs::MulAcc + for<'r> std::ops::MulAssign<&'r F> + Default


use num_traits::float::*; 
use std::cmp;

/// indexed value to keep track of position after sorting
#[derive(Copy, Clone)]
pub struct IndexedValue<F>(pub usize, pub F);

impl <F> IndexedValue<F> {
    pub fn new(idx:usize, val : F) -> Self {
        IndexedValue::<F>{0:idx, 1:val}
    }
} // end of impl block for IndexedValue


/// makes an ordering on Float by putting Nan at end of sort.
/// rust sorts in Increasing order so we reverse Greater and Less
pub(crate) fn decreasing_sort_nans_first<F : Float>(a: &IndexedValue<F>, b: &IndexedValue<F>) -> cmp::Ordering {
    match (a, b) {
        (x, y) if x.1.is_nan() && y.1.is_nan() => cmp::Ordering::Equal,
        (x, _) if x.1.is_nan() => cmp::Ordering::Greater,
        (_, y) if y.1.is_nan() => cmp::Ordering::Less,
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
/// To be used with Vec\<IndexedValue\<F\>\>::unstable_sort
impl <F:PartialOrd+PartialEq>  PartialOrd for IndexedValue<F> {
    fn partial_cmp(&self , other: &Self) -> Option<cmp::Ordering> {
        self.1.partial_cmp(&other.1)
    }
} // end of PartialOrd


mod tests {
    #[allow(unused)]
    use super::*;
    
 
    
    #[allow(unused)]
    use num_traits::{ToPrimitive, float::Float};
    
    #[allow(unused)]
    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }  

    #[test]
    fn check_sort_nan() {
        log_init_test();
        //
        let mut to_sort = Vec::<IndexedValue<f64>>::with_capacity(10);
        //
        to_sort.push(IndexedValue::<f64>::new(0, Float::nan()));
        to_sort.push(IndexedValue::<f64>::new(1, 3.));
        to_sort.push(IndexedValue::<f64>::new(2, Float::nan()));
        to_sort.push(IndexedValue::<f64>::new(3, 1.));
        to_sort.push(IndexedValue::<f64>::new(4, 5.));
        to_sort.push(IndexedValue::<f64>::new(5, Float::nan()));
        //
        to_sort.sort_unstable_by(decreasing_sort_nans_first);
        //
        for v in &to_sort {
            log::debug!(" (idx, va) : ({}, {})", v.0, v.1);
        }
        assert!((&to_sort[0].1 - 5.).abs() < 1.0E-10);
        assert!((&to_sort[1].1 - 3.).abs() < 1.0E-10);
        assert!((&to_sort[2].1 - 1.).abs() < 1.0E-10);
        assert!(&to_sort[3].1.is_nan());
        assert!(&to_sort[4].1.is_nan());
        assert!(&to_sort[5].1.is_nan());
    } // end of check_sort_nan

}  // end of mod tests