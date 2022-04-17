// compute degrees in and out from a csmat



use sprs::{CsMatI};


/// first component is in, second component is out!
pub struct Degree(pub u32, pub u32);

impl Degree {
    /// get degrre in 
    pub fn degree_in(&self) -> u32 {self.0}
    /// get degree out
    pub fn degree_out(&self) -> u32 {self.1}
}  // end of impl Degree



/// returns a vector of 2-uple consisting of degrees (in, out)
/// fist component is in, second component is out!
pub fn get_degrees<F>(csmat : &CsMatI<F, usize>) -> Vec<Degree> 
    where F : Copy + Default {
    //
    assert!(csmat.is_csr());
    //
    let (nb_row, _) = csmat.shape();
    let mut degrees = (0..nb_row).into_iter().map(|_| Degree(0u32,0u32)).collect::<Vec<Degree>>();
    //
    let mut iter = csmat.iter();
    while let Some((_val, (i,j))) = iter.next() {
        degrees[i].1 += 1;  // one more out for i
        degrees[j].0 += 1; // one more in for j
    }
    degrees
}  // end of get_degrees