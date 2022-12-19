//!

use anyhow::{anyhow};
use log::{log_enabled, Level};




use indxvec::Vecops;


// to get sorting with index as result
//

#[cfg_attr(doc, katexit::katexit)]
/// This structure store the decomposition of graph in maximal blocks of densest density.  
/// Increasing the size of a block would decrease its density. The decomposition is thus naturally defined.   
/// 
/// The blocks of vertices are of decreasing density. The exact decomposition is unique.
/// We provide here the approximate decomposition, which is the result of the function [approximate_decomposition](approximate_decomposition)
/// 
/// First we search maximal densest subsets $S_{i}$. There are deduced from the isotonic regression by a stability check.  
/// Then we get a diminishingly dense decomposition of the graph in blocks $B_{i}$.  
/// The blocks satisfy:
///  - $B_{i} \subset B_{i+1}$ 
///  - $B_{0}=\emptyset , B_{max}=V$ where $V$ is the set of vertices of G.
///  - each block must be stable.    
/// 
///  The blocks are defined by $B_{i} = B_{i-1} \cup S_{i}$
pub struct StableDecomposition {
    /// list of disjoint maximal densest subsets deduced from the isotonic regression.
    /// stable blocks filtered . Give for each point the $S_{i}$ to which the point belongs.
    /// Alternativeley s`[i`] contains the densest stable block to which node i belongs
    s : Vec<u32>,
    /// list of numblocks as given in s but sorted in increasing num
    index : Vec<usize>,
    /// for each block give its beginning position in index. 
    /// block\[0\] begins at 0, block\[1\] begins at index\[block_start\[1\]\]
    block_start : Vec<usize>,
} // end of struct StabeDecomposition


impl StableDecomposition {
    pub fn new(s : Vec<u32>) -> Self {
        log::info!("StableDecomposition::new");
        let index = s.mergesort_indexed().rank(true);
        log::debug!("sorted s : {}, {}", s[index[0]], s[index[1]]);
        let last_block = s[index[index.len()-1]] as usize;
        log::debug!("last block : {}", last_block);
        //
        let mut block_start  = Vec::<usize>::with_capacity(last_block);
        let mut current_block = 0;
        for i in 0..s.len() {
            log::debug!("i : {}, point : {}, current_block : {}", i, index[i], current_block);
            if i == 0 {
                assert_eq!(s[index[0]], 0);
                block_start.push(i);
            }
            else if s[index[i]]  > current_block {
                log::debug!("bloc : {} has size : {}", current_block, i - block_start[current_block as usize]);
                block_start.push(i);
                current_block = s[index[i]];
            }
            else if i == s.len() - 1 {
                log::debug!("bloc : {} has size : {}", current_block, s.len() - block_start[current_block as usize]);
            }
        }
        //
        if log_enabled!(Level::Debug) {
            log::debug!(" indexes : {:?}", index);
            log::debug!(" block start : {:?}", block_start);
        }
        //
        StableDecomposition{s, index, block_start}
    }  // end of new StableDecomposition


    /// get number of points in a block
    pub fn get_nbpoints_in_block(&self, blocknum : usize) -> Result<usize,()> {
        let size = self.block_start.len();
        //
        if blocknum >= size {
            return Err(());
        }
        else if blocknum == size - 1 {
            return Ok(self.s.len() - self.block_start[blocknum]);
        }
        else {
            return Ok(self.block_start[blocknum+1] - self.block_start[blocknum]);
        }
        //
    } // end of get_nbpoints_in_block



    /// get densest block num for a node
    pub fn get_densest_block(&self, node: usize) -> Result<usize,()> { 
        if node < self.s.len() {
            Ok(self.s[node] as usize)
        }
        else {
            Err(())
        }
    }  // end of get_densest_block

    /// returns the number of blocks of the decomposition
    pub fn get_nb_blocks(&self) -> usize { 
        if self.index.len() > 0 { 1 + self.s[self.index[self.index.len()-1]] as usize } 
        else {0usize} 
    }

    /// returns the points in a given block
    pub fn get_block_points(&self, blocknum : usize) -> Result<Vec<usize>, anyhow::Error> {
        //
        log::debug!("\n  in get_block_points, blocnum : {}", blocknum);
        //
        if blocknum >= self.block_start.len() {
            return Err(anyhow!("too large num of block"));
        }
        let nb_block = self.block_start.len();
        let mut pt_list = Vec::<usize>::new();
        let start = self.block_start[blocknum];
        let end = if blocknum < nb_block - 1 { self.block_start[blocknum+1]} else { self.index.len() };
        log::debug!("bloc start in index : {}, end (excluded) : {}", start, end);
        for p in start..end {
            pt_list.push(self.index[p]);
        }
        if log_enabled!(Level::Debug) {
            log::debug!(" indexes : {:?}", self.index);
            log::debug!(" block start : {:?}", start);
        }
        Ok(pt_list)
    } // end of get_block_points
} // end of impl StableDecomposition


