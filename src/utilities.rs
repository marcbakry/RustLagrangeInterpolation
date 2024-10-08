extern crate num_traits;
extern crate num;

use num::{cast::AsPrimitive,complex::*};
use num_traits::{Float,zero,NumCast,one};
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

pub fn check_duplicate<T: LagRealTrait>(xa: &Vec<T>) {
    let mut xac = xa.clone();
    xac.sort_by(|a,b| a.partial_cmp(b).unwrap());
    let _ = (0..xac.len()-1).map(|i| {let val = (xac[i+1]-xac[i]).abs().as_(); if val < 1e-10 { panic!("'xa' has duplicated entries");
    } else {return val;}}).collect::<Vec<f64>>();
}

pub fn gauss_chebyshev_nodes<T: LagRealTrait>(n: &usize, a: &T, b: &T) -> Vec<T> where i32: AsPrimitive<T> {
    (0..*n).map(|i| {
        let x = -T::cos((T::acos(-one::<T>())*(i as i32).as_())/((*n-1) as i32).as_());
        return rescale_range(a, b, &x);
    }).collect::<Vec<_>>()
}

fn rescale_range<T: LagRealTrait>(a: &T, b: &T, x: &T) -> T where i32: AsPrimitive<T> {
    return ((*b-*a)*(*x) + *a + *b)/(2.as_());
}

// TESTS
#[cfg(test)]
pub mod utilities_tests {
    use super::*;

    #[test]
    pub fn is_midpoint() {
        let n=5;
        let stp = 1.0/((n-1) as f64);
        let x = (0..n).map(|i| i as f64*stp).collect::<Vec<f64>>();
        let x_mid = midpoints(&x);
        // 
        assert_eq!(x.len()-1,x_mid.len());
        // 
        let _ = (0..x_mid.len()).map(|i| {
            let val = (x_mid[i] - (x[i+1]+x[i])/2.0).abs();
            if val > 1e-14 {
                panic!("midpoints() failed");
            } else {
                true
            }
        }).collect::<Vec<_>>();
    }

    #[test]
    pub fn working_partial_sum() {
        let xa = vec![1.0,2.0,3.0,4.0] ;
        let x = 1.5;
        let j = 2;

        let val = partial_sum(&xa, x, j);
        assert!(Float::abs(val+0.4) < 1e-14);
    }

    #[test]
    pub fn argsort_indices_are_sorted() {
        let x = vec![1.1,-2.0,0.4,3.0,std::f64::consts::PI];
        let indices = argsort(&x);

        assert_eq!(indices,vec![1,2,0,3,4]);
    }

    #[test]
    #[should_panic]
    pub fn no_duplicates() {
        let x = vec![1.0,2.0,1.0,-3.0];
        check_duplicate(&x);
    }
}