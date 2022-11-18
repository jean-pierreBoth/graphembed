#![allow(unused)]
//! Isotonic regression with PAVA algorithm

// This file is modified from the crate pav_regression
// Added following modifications:
// - avoid reallocations of points to be dispatched
// - genericity over f32, f64
// - added struct BlockPoint that keep track of index of points in blocks through merge operations
// - methods to get contents of each block



use anyhow::{anyhow};

use ordered_float::OrderedFloat;
use num_traits::float::Float;

use num_traits::{NumAssign,FromPrimitive};
use std::iter::{Sum, Product};
use std::ops::{AddAssign};
use std::fmt::{Debug, Display};

use indxvec::Vecops;

use std::cell::RefCell;

const EPSIL : f64 = 1.0E-6;

/// Isotonic regression can be done in either mode
#[derive(Debug, PartialEq, Copy, Clone)]
enum Direction {
    Ascending,
    Descending,
}
/// A point in 2D cartesian space
#[derive(Debug, PartialEq, Copy, Clone)]
pub struct Point<T:Float+Debug> {
    x: T,
    y: T,
    weight: T,
}

/// default point. zero weight so do not count
impl <T> Default for Point<T> where T : Float + Debug {
    fn default() -> Self {
        Point{ x : T::zero(), y : T::zero(), weight : T::zero()}
    }
} // end of default for Point



impl <T> Point<T> 
    where  T : Float + Debug + std::ops::AddAssign {
    /// Create a new Point
    pub fn new(x: T, y: T) -> Point<T> {
        Point { x, y, weight: T::from(1.0).unwrap() }
    }

    /// Create a new Point with a specified weight
    pub fn new_with_weight(x: T, y: T, weight: T) -> Point<T> {
        Point { x, y, weight }
    }

    /// The x position of the point
    pub fn x(&self) -> T {
        self.x
    }

    /// The y position of the point
    pub fn y(&self) -> T {
        self.y
    }

    /// The weight of the point (initially 1.0)
    pub fn weight(&self) -> T {
        self.weight
    }

    // useful to merge centroids
    fn merge_with(&mut self, other: &Point<T>) {
        self.x = ((self.x * self.weight) + (other.x * other.weight)) / (self.weight + other.weight);
        self.y = ((self.y * self.weight) + (other.y * other.weight)) / (self.weight + other.weight);
        self.weight += other.weight;
    }
}


/// ordering with respect to x for sorting methods. But centroids are compared with respect to y!
impl <T:Float+ Debug> PartialOrd for Point<T> {
    fn partial_cmp(&self, other: &Point<T>) -> Option<std::cmp::Ordering> {
        self.x.partial_cmp(&other.x)
    }
} // end of impl PartialOrd for Point<T> 


fn interpolate_two_points<T>(a: &Point<T>, b: &Point<T>, at_x: &T) -> T  
    where T : Float + Debug {
    let prop = (*at_x - (a.x)) / (b.x - a.x);
    (b.y - a.y) * prop + a.y
}


//==========================================================================================================

// This structure does the merging.
/// To store a block of iso values points in isotonic regression
/// This structure stores the index(es) of the original points in the block
#[derive(Debug, Copy, Clone)]
pub struct BlockPoint<'a, T:Float + Debug> {
    /// sorting direction, TODO do we need it ?
    direction : Direction,
    /// unsorted points,   
    points : &'a [Point<T>],
    /// so that i -> points[sorted_index[i]] is sorted according to direction
    index : &'a[usize],
    /// first index in sorted index. first is in block. So the block is [first, last[
    first : usize,
    /// last index in sorted index, last is outside block
    last : usize,
    //
    centroid : Point<T>,
} // end of BlockPoint


impl <'a, T> BlockPoint<'a, T>  
    where T : Float + std::ops::DivAssign + std::ops::AddAssign + std::ops::DivAssign + Debug {
    
    fn new(direction: Direction, points : &'a Vec<Point<T>>, index : &'a [usize], first : usize, last : usize) -> Self {
        BlockPoint{direction, points, index, first , last, centroid : Point::<T>::default()}
    }

    // creation of block from a point
    fn new_from_point(direction: Direction, points : &'a [Point<T>], index : &'a [usize], idx : usize) -> Self {
        let centroid = points[index[idx]].clone();
        BlockPoint{direction, points, index,  first : idx , last : idx+1, centroid}
    }


    /// merge two contiguous BlockPoint
    fn merge(&mut self, other : &BlockPoint<'a, T>) ->  Result<(), anyhow::Error> {
        log::debug!("entering block merge self, {} {} other :  {} {} ", self.first, self.last, other.first, other.last);
        // check contiguity
        if self.last == other.first {
            self.last = other.last;
        }
        else if self.first == other.last {
            self.first = other.first;
        }
        else {
            log::error!("not contiguous blocks");
            return Err(anyhow!("not contiguous blocks"));                    
        }
        // update centroid of blocks
        self.centroid.merge_with(&other.centroid);
        //
        log::debug!("exiting block merge self, {} {}", self.first, self.last);
        //
        return Ok(());
    } // end of merge


    // return centroid
    pub fn get_centroid(&self) ->  Point<T> {
        self.centroid
    }

    /// get first index of block in the Direction ordering
    fn get_first_index(&self) -> usize {
        self.first
    }

    /// get last index of block in the Direction ordering
    fn get_last_index(&self) -> usize {
        self.last
    }

    // return true if self is consistently ordrered with other, means self < other in ascending self > other in descending
    fn is_ordered(&self, other : &BlockPoint<T>) -> bool {
        assert_eq!(self.direction, other.direction);
        let ordered = match self.direction {
            Direction::Ascending => {
                if self.centroid.y < other.centroid.y { true } else { false}
            },
            Direction::Descending => {
                if self.centroid.y > other.centroid.y { true } else { false}
            },
        };
        ordered
    } // end of is_ordered

    // debug utility
    #[allow(unused)]
    fn dump(&self) {
        log::debug!("block dump");
        println!("first last centroid : {}  {}   {:?}", self.first, self.last, self.centroid);
        println!("index : {:?}", self.index);
    }
} // end of impl BlockPoint


impl <'a, T:Float + Debug> PartialEq for BlockPoint<'a,T> {
    fn eq(&self, other: &BlockPoint<T>) -> bool {
        self.centroid.eq(&other.centroid)
    }
} // end of impl PartialOrd for BlockPoint<T> 


/// ordering with respect to x (!!) for sorting methods. But centroids are classified  with respect to y!
impl <'a, T:Float + Debug> PartialOrd for BlockPoint<'a,T> {
    fn partial_cmp(&self, other: &BlockPoint<T>) -> Option<std::cmp::Ordering> {
        self.centroid.x.partial_cmp(&other.centroid.x)
    }
} // end of impl PartialOrd for BlockPoint<T> 



fn interpolate_two_blockpoints<T>(a: &BlockPoint<T>, b: &BlockPoint<T>, at_x: &T) -> T  
    where T : Float + Debug {
    let prop = (*at_x - (a.centroid.x)) / (b.centroid.x - a.centroid.x);
    (b.centroid.y - a.centroid.y) * prop + a.centroid.y
}

//==========================================================================================================

/// A vector of points forming an isotonic regression, along with the
/// centroid point of the original set.

#[derive(Debug)]
pub struct IsotonicRegression<'a, T:Float + Debug + 'static> {
    direction : Direction,
    /// points, unsorted,
    points: &'a[Point<T>],
    /// index for sorting points according to direction
    index : Vec<usize>,
    // blocks. RefCell makes call to do_isotonic without &mut!
    blocks : RefCell<Vec<BlockPoint<'a, T>>>,
    // global centroid
    centroid_point: Point<T>,
} // end of struct IsotonicRegression


impl <'a, T> IsotonicRegression<'a, T> 
    where T : Float + std::iter::Sum + FromPrimitive + std::ops::AddAssign + std::ops::DivAssign + Debug + 'static {
    /// Find an ascending isotonic regression from a set of points
    pub fn new_ascending(points: &[Point<T>]) -> IsotonicRegression<T> {
        IsotonicRegression::new(points, Direction::Ascending)
    }

    /// Find a descending isotonic regression from a set of points
    pub fn new_descending(points: &[Point<T>]) -> IsotonicRegression<T> {
        IsotonicRegression::new(points, Direction::Descending)
    }

    fn new(points: &'a [Point<T>], direction: Direction) -> IsotonicRegression<T> {
        assert!(points.len() > 0, "points is empty, can't create regression");
        let point_count: T = points.iter().map(|p| p.weight).sum();
        let mut sum_x: T = T::from(0.0).unwrap();
        let mut sum_y: T = T::from(0.0).unwrap();
        for point in points {
            sum_x += point.x * point.weight;
            sum_y += point.y * point.weight;
        }
        // get a index for access to sorted values
        let mut index = points.mergesort_indexed();
        let blocks = Vec::<BlockPoint::<'a, T>>::new();
        IsotonicRegression {
            direction,
            points: points,
            index : index,
            blocks : RefCell::new(blocks),
            centroid_point: Point::new(sum_x / point_count, sum_y / point_count),
        }
    } // end of new 


    // retrieve the block for a point
    pub fn find_block(&self, at_x : T) -> &BlockPoint<'a,T> {
        std::panic!("not yet implemented")
    } // end of find_block


    /// returns the index of orignal points that are in this block
    pub fn get_point_index(&self) -> Vec<usize> {
        std::panic!("not yet implemented")
    } // end of get_point_index


    /// Find the _y_ point at position `at_x`
    pub fn interpolate<'b:'a>(&'b self, at_x: T) -> T 
        where T : Float {
        //
        log::debug!("interpolate nb blocks = {}", self.blocks.borrow().len());
        if self.blocks.borrow().len() == 0 {
            log::info!("uninitialized regression, running do_isotonic");
            self.do_isotonic();
        }
        let blocks =  self.blocks.borrow();
        //
        if blocks.len() == 1 {
            return blocks[0].centroid.y;
        } else {
            let pos = blocks.binary_search_by_key(&OrderedFloat(at_x), |p| OrderedFloat(p.centroid.x));
            return match pos {
                Ok(ix) => blocks[ix].centroid.y,
                Err(ix) => {
                    if ix < 1 {
                        interpolate_two_points(
                            &blocks.first().unwrap().centroid,
                            &self.centroid_point,
                            &at_x,
                        )
                    } else if ix >= self.points.len() {
                        interpolate_two_points(
                            &self.centroid_point,
                            &blocks.last().unwrap().centroid,
                            &at_x,
                        )
                    } else {
                        interpolate_two_points(&blocks[ix - 1].centroid, &blocks[ix].centroid, &at_x)
                    }
                }
            };
        }
    }

    /// Retrieve the points that make up the isotonic regression
    pub fn get_points(&self) -> &[Point<T>] {
        &self.points
    }

    /// Retrieve the mean point of the original point set+
    pub fn get_centroid_point(&self) -> &Point<T> {
        &self.centroid_point
    }

    //
    fn do_isotonic<'b:'a>(&'b self)-> Result<(), anyhow::Error>  {
        //
        if self.blocks.borrow().len() != 0 {
            return Err(anyhow!("regression already done!"));
        }
        //        
        let epsil = T::from(EPSIL).unwrap();
        // we must ensure that there is one initial block point by x coordinate, to guarantee consistent block merge
        let mut blocks: Vec<RefCell<BlockPoint<T>>>  = Vec::new(); 
        for i in 0..self.points.len() {
            let new_block = BlockPoint::<T>::new_from_point(self.direction, self.points, &self.index, i);
            assert!(i==0 || (i > 0 && self.points[self.index[i]].x > self.points[self.index[i-1]].x));
            if i== 0 || ( i>0 && self.points[self.index[i]].x - self.points[self.index[i-1]].x > epsil) {
                blocks.push(RefCell::new(new_block));
            }
            else {
                log::debug!("merging two equal points {:#?} {:#?}", self.points[self.index[i-1]].x, self.points[self.index[i]].x);
                let last_block = blocks.pop().unwrap();
                last_block.borrow_mut().merge(&new_block);
                blocks.push(last_block);
            }
        }
        log::info!("nb blocks with different x : {}", blocks.len());
        // we merge blocks as soon there is an ordering violation
        // We scan points according to index. The test of block creation must depend on direction.
        let mut iso_blocks : Vec<RefCell<BlockPoint<T>>>  = Vec::new(); 
        // TODO possibly we get cache problem and we need to work on a cloned sorted point array? at memory expense
        for i in 0..blocks.len() {
            // check violation with preceding block
            let mut inserted = false;
            let new_iso = blocks[i].clone();
            while !inserted {
                if iso_blocks.is_empty() || iso_blocks.last().unwrap().borrow().is_ordered(&new_iso.borrow()) {
                    log::debug!("inserting block, centroid : {:?} ", new_iso.borrow().centroid);
                    iso_blocks.push(new_iso.clone());
                    inserted = true;
                }
                else {
                    let previous = iso_blocks.pop().unwrap();
                    new_iso.borrow_mut().merge(&previous.borrow()).unwrap();
                }
            }
        } // end of for on blocks
        //
        log::info!("after final merge nb blocks = {}", iso_blocks.len());
        // transfer to raw blocks
        let final_blocks : Vec<BlockPoint<T>> = iso_blocks.into_iter().map(|obj|  obj.into_inner() ).collect();
        *self.blocks.borrow_mut() = final_blocks;
        if log::log_enabled!(log::Level::Debug) {
            self.print_blocks();
        }
        //
        return Ok(());
    }  // end of do_isotonic

    // check contiguity and order
    fn check_blocks(&self) -> bool {
        return false;
    }

    // debugging method
    fn print_blocks(&self) {
        log::debug!("dump of blocks");
        let blocks = self.blocks.borrow();
        for i in 0..blocks.len() {
            blocks[i].dump();
        }
    }
} // end of impl  IsotonicRegression<'a, T> 




fn isotonic<T>(points: &[Point<T>], direction: Direction) -> Vec<Point<T>> 
    where T : Float + Debug + AddAssign {
    let mut merged_points: Vec<Point<T>> = match direction {
        Direction::Ascending => points.iter().copied().collect(),
        Direction::Descending => points.iter().map(|p| Point { y: -p.y, ..*p }).collect(),
    };

    merged_points.sort_by_key(|point| OrderedFloat(point.x));

    let mut iso_points: Vec<Point<T>> = Vec::new();
    for point in &mut merged_points.iter() {
        if iso_points.is_empty() || (point.y > iso_points.last().unwrap().y) {
            iso_points.push(*point)
        } else {
            let mut new_point = *point;
            loop {
                if iso_points.is_empty() || (iso_points.last().unwrap().y < (new_point).y) {
                    iso_points.push(new_point);
                    break;
                } else {
                    let last_to_repl = iso_points.pop();
                    new_point.merge_with(&last_to_repl.unwrap());
                }
            }
        }
    }

    return match direction {
        Direction::Ascending => iso_points,
        Direction::Descending => iso_points.iter().map(|p| Point { y: -p.y, ..*p }).collect(),
    };
}




#[cfg(test)]
mod tests {
    use super::*;

    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }  


    #[test]
    fn usage_example() {
        //
        log_init_test();
        //
        let points = &[
            Point::<f64>::new(0.0, 1.0),
            Point::<f64>::new(1.0, 2.0),
            Point::<f64>::new(2.0, 1.5),
        ];
        let regression = IsotonicRegression::new_ascending(points);
        regression.do_isotonic().unwrap();
        assert_eq!(
            regression.interpolate(1.5), 1.75
        );
    }

    #[test]
    fn isotonic_no_points() {
    //
    log_init_test();
    //
    assert_eq!(isotonic(&[] as &[Point<f64>; 0], Direction::Ascending).is_empty(), true);
    }

    #[test]
    fn isotonic_one_point() {
        assert_eq!(
            isotonic(&[Point::<f64>::new(1.0, 2.0)], Direction::Ascending)
                .pop()
                .unwrap(),
            Point::<f64>::new(1.0, 2.0)
        );
    }

    #[test]
    fn isotonic_simple_merge() {
        //
        log_init_test();
        //        
        assert_eq!(
            isotonic(
                &[Point::<f64>::new(1.0, 2.0), Point::<f64>::new(2.0, 0.0)],
                Direction::Ascending
            )
            .pop()
            .unwrap(),
            Point::new_with_weight(1.5, 1.0, 2.0)
        );
    }

    #[test]
    fn isotonic_one_not_merged() {
        //
        log_init_test();
        // 
        assert_eq!(
            isotonic(
                &[
                    Point::new(0.5, -0.5),
                    Point::new(1.0, 2.0),
                    Point::new(2.0, 0.0),
                ],
                Direction::Ascending
            ),
            [Point::new(0.5, -0.5), Point::new_with_weight(1.5, 1.0, 2.0)]
        );
    }

    #[test]
    fn isotonic_merge_three() {
        //
        log_init_test();
        // 
        assert_eq!(
            isotonic(
                &[
                    Point::new(0.0, 1.0),
                    Point::new(1.0, 2.0),
                    Point::new(2.0, -1.0),
                ],
                Direction::Ascending
            ),
            [Point::new_with_weight(1.0, 2.0 / 3.0, 3.0)]
        );
    }

    #[test]
    fn test_interpolate() {
        //
        log_init_test();
        // 
        let points = [Point::new(1.0, 5.0), Point::new(2.0, 7.0)];
        let regression =
            IsotonicRegression::new_ascending(&points);
        assert!((regression.interpolate(1.5) - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_isotonic_ascending() {
        //
        log_init_test();
        // 
        let points = &[
            Point::new(0.0, 1.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, -1.0),
        ];

        let regression = IsotonicRegression::new_ascending(points);
        assert_eq!(
            regression.get_points(),
            &[Point::new_with_weight(
                (0.0 + 1.0 + 2.0) / 3.0,
                (1.0 + 2.0 - 1.0) / 3.0,
                3.0
            )]
        )
    }

    #[test]
    fn test_isotonic_descending() {
        //
        log_init_test();
        // 
        let points = &[
            Point::new(0.0, -1.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, 1.0),
        ];
        let regression = IsotonicRegression::new_descending(points);
        assert_eq!(
            regression.get_points(),
            &[Point::new_with_weight(1.0, 2.0 / 3.0, 3.0)]
        )
    }

    #[test]
    fn test_descending_interpolation() {
        //
        log_init_test();
        // 
        let points = [
            Point::new(0.0, 3.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, 1.0),
        ];
        let regression = IsotonicRegression::new_descending(&points);
        assert_eq!(regression.interpolate(0.5), 2.5);
    }

    #[test]
    fn test_single_point_regression() {
        //
        log_init_test();
        // 
        let points = [Point::new(1.0, 3.0)];
        let regression = IsotonicRegression::new_ascending(&points);
        assert_eq!(regression.interpolate(0.0), 3.0);
    }

    #[test]
    fn test_point_accessors() {
        //
        log_init_test();
        // 
        let point = Point { x: 1.0, y: 2.0 , weight : 3.0};
        assert_eq!(point.x(), 1.0);
        assert_eq!(point.y(), 2.0);
        assert_eq!(point.weight(), 3.0);
    }
}
