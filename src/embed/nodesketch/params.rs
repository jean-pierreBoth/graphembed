//! The module defines parameters for nodesketch embedding.
//!
//! As weight given to an edge depends exponentially on the number of hops we made from origin node,
//! The decay must be related to the number of hops (iterations) asked around a node.

#[derive(Debug, Copy, Clone)]
pub struct NodeSketchParams {
    /// size of the sketch or number of hashed values used for representing a node.   
    /// In fact it is the dimension of the embeding.
    pub sketch_size: usize,
    /// exponential decay coefficient for reducing weight of a neighbour at each hop.
    pub decay: f64,
    /// number of iterations (i.e of hops around a node)
    pub nb_iter: usize,
    /// symetric mode or not.
    pub symetric: bool,
    /// parallel mode
    pub parallel: bool,
} // end of NodeSketchParams

impl NodeSketchParams {
    #[cfg_attr(doc, katexit::katexit)]
    ///
    /// Generally only a few hops are necessary around a node to characterize neighborhood (and so probable missing edges ni validation).
    /// The weight $w$ of an edge at a distance $nbhops$ from the node around which we work is given by   $$ w = decay^{nbhops}$$      
    /// So decay and nb_iter (nb_hops) must be chosen so that the edge weight is still significant.
    ///
    pub fn new(
        sketch_size: usize,
        decay: f64,
        nb_iter: usize,
        symetric: bool,
        parallel: bool,
    ) -> Self {
        NodeSketchParams {
            sketch_size,
            decay,
            nb_iter,
            symetric,
            parallel,
        }
    }
    //
    pub fn get_decay_weight(self) -> f64 {
        self.decay
    }

    //
    pub fn get_nb_iter(&self) -> usize {
        self.nb_iter
    }

    //
    pub fn is_symetric(&self) -> bool {
        self.symetric
    }

    //
    pub fn get_parallel(&self) -> bool {
        self.parallel
    }

    //
    pub fn get_sketch_size(&self) -> usize {
        self.sketch_size
    }

    /// useful to set flag received from argument related to datafile reading
    /// or for getting asymetric embedding even with symetric data to check coherence/stability
    /// but running in symetric mode with asymetric graph is checked against.
    pub fn set_symetry(&mut self, symetry: bool) {
        self.symetric = symetry
    }
} // end of NodeSketchParams
