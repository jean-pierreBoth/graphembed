//! Sketching Parameters
//! 
//! 
//! 
#[derive(Debug, Copy, Clone)]
pub struct SketchParams {
    /// size of the sketch or number of hashed values used for representing a node.   
    /// In fact it is the dimension of the embeding.
    pub sketch_size: usize,    
    /// exponential decay coefficient for reducing weight of a neighbour at each hop.
    pub decay : f64,
    /// number of iterations (i.e of hops around a node)
    pub nb_iter : usize,
    /// symetric mode or not.
    pub symetric : bool,
    /// parallel mode
    pub parallel : bool,
} // end of NodeSketchParams


impl SketchParams {

    pub fn new(sketch_size: usize, decay : f64, nb_iter : usize, symetric: bool, parallel : bool) -> Self {
        SketchParams{sketch_size, decay, nb_iter, symetric, parallel}
    }
    /// 
    pub fn get_decay_weight(self) -> f64 { self.decay }

    /// 
    pub fn get_nb_iter(&self) -> usize { self.nb_iter }

    ///
    pub fn is_symetric(&self) -> bool { self.symetric }

    ///
    pub fn get_parallel(&self) -> bool { self.parallel }

    ///
    pub fn get_sketch_size(&self) -> usize { self.sketch_size}

    /// useful to set flag received from argument related to datafile reading
    /// or for getting asymetric embedding even with symetric data to check coherence/stability
    /// but running in symetric mode with asymetric graph is checked against.
    pub fn set_symetry(&mut self, symetry : bool) { self.symetric = symetry}

} // end of SketchParams