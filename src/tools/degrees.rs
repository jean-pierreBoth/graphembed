//! compute degrees in and out from a csmat



use sprs::{CsMatI};


/// first component is in, second component is out!
#[derive(Copy,Clone, Debug)]
pub struct Degree {
    pub d_in : u32, 
    pub d_out : u32,
}

impl Degree {
    fn new(d_in : u32, d_out : u32) -> Self { Degree{d_in, d_out} }
    /// get degrre in 
    pub fn degree_in(&self) -> u32 {self.d_in}
    /// get degree out
    pub fn degree_out(&self) -> u32 {self.d_out}
}  // end of impl Degree



/// returns a vector of 2-uple consisting of degrees (in, out)
/// fist component is in, second component is out!
/// Self loops are not taken into account as the objective of this function 
/// to be able to delete edge in AUC link prediction and avoiding making disconnected nodes
pub(crate) fn get_degrees<F>(csmat : &CsMatI<F, usize>) -> Vec<Degree> 
    where F : Copy + Default {
    //
    assert!(csmat.is_csr());
    //
    let (nb_row, _) = csmat.shape();
    let mut degrees = (0..nb_row).into_iter().map(|_| Degree::new( 0, 0)).collect::<Vec<Degree>>();
    //
    let mut iter = csmat.iter();
    while let Some((_val, (i,j))) = iter.next() {
        if i!=j {
            degrees[i].d_out += 1;  // one more out for i
            degrees[j].d_in += 1; // one more in for j
        }
    }
    degrees
}  // end of get_degrees