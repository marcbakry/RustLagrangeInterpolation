extern crate num_traits;
extern crate num;

use num::{cast::AsPrimitive,complex::*};
use num_traits::{Float,zero,NumCast};
use std::cmp::PartialOrd;
use std::fmt::Display;
use std::ops::AddAssign;

pub trait LagRealTrait: Copy+Float+AsPrimitive<f64>+PartialOrd+Display {}
impl<T> LagRealTrait for T where T: Copy+Float+AsPrimitive<f64>+PartialOrd+Display {}

pub trait LagComplexTrait: NumCast + AddAssign + ComplexFloat {}
impl<T> LagComplexTrait for T where T: NumCast + AddAssign + ComplexFloat, <T as ComplexFloat>::Real: Float {}

pub fn midpoints<T: LagRealTrait>(x: &Vec<T>)->Vec<T> where i32: AsPrimitive<T> {
    (0..(x.len()-1)).map(|i| (x[i] + x[i+1])/(2.as_())).collect()
}

pub fn argsort<T: PartialOrd>(x: &Vec<T>) -> Vec<usize> {
    let mut indices = (0..x.len()).collect::<Vec<usize>>();
    indices.sort_by(|a,b| x[*a].partial_cmp(&x[*b]).unwrap());
    // 
    return indices;
}

pub fn partial_sum<T: LagRealTrait>(xa: &Vec<T>, x: T, j: usize) -> T {
    (0..xa.len()).map(|i| {
        if i!=j {
            return num_traits::one::<T>()/(x-xa[i]);
        } else {
            return num_traits::zero::<T>();
        }
    }).fold(zero::<T>(), |acc,e| acc+e)
}