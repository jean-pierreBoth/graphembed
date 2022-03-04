//! defines parameters for nodesketch embedding


#[derive(Debug, Copy, Clone)]
pub struct NodeSketchParams {
    /// size of the sketch
    pub sketch_size: usize,    
    /// exponential decay coefficient for reducing weight of 
    pub decay : f64,
    /// number of iterations (i.e of hops around a node)
    pub nb_iter : usize,
    /// parallel mode
    pub parallel : bool,
} // end of NodeSketchParams


impl NodeSketchParams {

    pub fn new(sketch_size: usize, decay : f64, nb_iter : usize, parallel : bool) -> Self {
        NodeSketchParams{sketch_size, decay, nb_iter, parallel}
    }

    /// 
    pub fn get_decay_weight(self) -> f64 { self.decay }

    /// 
    pub fn get_nb_iter(&self) -> usize { self.nb_iter }

    ///
    pub fn get_parallel(&self) -> bool { self.parallel }

    ///
    pub fn get_sketch_size(&self) -> usize { self.sketch_size}

} // end of NodeSketchParams