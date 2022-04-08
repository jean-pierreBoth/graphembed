//! Validation parameters


/// A structure describing validation strategy.
/// At present time only link prediction is implemented
/// 
//  TODO nodelabeling
#[derive(Copy,Clone,Debug)]
pub struct ValidationParams {
    /// The fraction of edges to skip when construction the train set
    delete_fraction : f64,
    /// the number of pass to run
    nbpass : usize,
    /// We must know if graph is symetric as we will have to delete edges in a coherent way!
    symetric : bool,
} // end of ValidationParams



impl ValidationParams {
    pub fn new(delete_fraction : f64, nbpass : usize, symetric : bool) -> Self {
        ValidationParams{delete_fraction, nbpass, symetric}
    }

    /// number of pass in validation
    pub fn get_nbpass(&self) -> usize { self.nbpass}

    /// fraction to skip in train test
    pub fn get_delete_fraction(&self) -> f64 { self.delete_fraction}

    /// 
    pub fn is_symetric(&self) -> bool { self.symetric}
}  // end of ValidationParams
