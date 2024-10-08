extern crate num_traits;

use num_traits::{zero,AsPrimitive};
use std::fmt::{Debug,Display,Formatter,Result};

use super::utilities::*;
use super::lag2_utilities::*;

#[derive(Debug,Clone)]
pub struct Lagrange2dInterpolator<T,U> {
    xa: Vec<T>,
    ya: Vec<U>,
    diff1_order: usize,
    diff2_order: usize
}