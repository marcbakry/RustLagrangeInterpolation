//! Module containing useful functions which do not fit particularly in more 
//! specialized modules (equally spaced nodes, barycentric weights computation, etc.).
extern crate num_traits;
extern crate num;

use num::{cast::AsPrimitive,complex::*};
use num_traits::{Float,zero,one};
use std::cmp::PartialOrd;
use std::fmt::Display;
use std::ops::{Add, Sub, Mul, Div, AddAssign, SubAssign, DivAssign, MulAssign};

/// Trait for all basic arithmetic operations and assignments
pub trait LagBasicArithmetic<U=Self>: Add<U,Output=Self>+Sub<U,Output=Self>+Div<U,Output=Self>+Mul<U,Output=Self>+AddAssign<U>+SubAssign<U>+MulAssign<U>+DivAssign<U> {}
impl <T,U> LagBasicArithmetic<U> for T where T: Add<U,Output=Self>+Sub<U,Output=Self>+Div<U,Output=Self>+Mul<U,Output=Self>+AddAssign<U>+SubAssign<U>+MulAssign<U>+DivAssign<U> {}

/// Trait allowing the support of floating point data
pub trait LagRealTrait: 'static + Float + Copy+AsPrimitive<f64>+PartialOrd+Display+Send+Sync+std::iter::Product+std::iter::Sum + LagBasicArithmetic {}
impl<T> LagRealTrait for T where T: 'static + Float + Copy+AsPrimitive<f64>+PartialOrd+Display+Send+Sync+std::iter::Product+std::iter::Sum + LagBasicArithmetic {}

pub trait LagComplexTrait: 'static + ComplexFloat + Copy + Display + Send + Sync + std::iter::Product + std::iter::Sum + LagBasicArithmetic {}
impl<T> LagComplexTrait for T where T: 'static + ComplexFloat + Copy + Display + Send + Sync + std::iter::Product + std::iter::Sum + LagBasicArithmetic {}


/// Returns a vector containing the middle point of the input vector
/// 
/// #Example
/// 
/// ```
/// let a = vec![1.0,2.0,3.0];
/// let a_mid = midpoints(&a); // contains [1.5,2.5]
/// ```
pub fn midpoints<T: LagRealTrait>(x: &Vec<T>)->Vec<T> where i32: AsPrimitive<T> {
    (0..(x.len()-1)).map(|i| (x[i] + x[i+1])/(2.as_())).collect()
}

/// Returns the indices sorting the real input in ascending order
/// 
/// # Example
/// 
/// ```
/// let a = vec![2.0,3.0,1.0];
/// let idx = argsort(&a); // contains [2,0,1]
/// ```
pub fn argsort<T: PartialOrd>(x: &Vec<T>) -> Vec<usize> {
    let mut indices = (0..x.len()).collect::<Vec<usize>>();
    indices.sort_by(|a,b| x[*a].partial_cmp(&x[*b]).unwrap());
    // 
    return indices;
}

/// Routine computing the following partial sum `\sum_{i=1, i\neq j}^{n}{\frac{1}{x - xa_i}}`
/// for an input vector `xa`` with length  `n`.
/// 
/// # Example
/// 
/// ```
/// let xa = [1.0,2.0,3.0];
/// let j = 1;
/// let x = 1.5;
/// let p = partial_sum(&xa,&x,j); // contains 1.0/(x-xa[0])+1.0/(x-xa[2])
/// ```
pub fn partial_sum<T: LagRealTrait>(xa: &Vec<T>, x: &T, j: usize) -> T {
    (0..xa.len()).map(|i| {
        if i!=j {
            return num_traits::one::<T>()/(*x-xa[i]);
        } else {
            return num_traits::zero::<T>();
        }
    }).fold(zero::<T>(), |acc,e| acc+e)
}

/// Checks for duplicated entries in the input vector
/// 
/// # Panics
/// 
/// This function will panic if two nodes are identical.
/// 
/// # Example
/// 
/// ```
/// let a = [1.0,1.0,2.0];
/// check_duplicate(&a); // should panic
/// ```
pub fn check_duplicate<T: LagRealTrait>(xa: &Vec<T>) {
    let mut xac = xa.clone();
    xac.sort_by(|a,b| a.partial_cmp(b).unwrap());
    let _ = (0..xac.len()-1).map(|i| {let val = Float::abs(xac[i+1]-xac[i]).as_(); if val < 1e-10 { panic!("'xa' has duplicated entries");
    } else {return val;}}).collect::<Vec<f64>>();
}

/// Computes the n Gauss-Chebyshev nodes on the interval `[a,b]` such that `x_i = -cos(pi i/(n-1))`
/// 
/// # Example
/// 
/// ```
/// let (a,b,n) = (0.0,1.0,3);
/// let x = gauss_chebyshev_nodes(&n,&a,&b); // should contain [0.0,0.5,1.0]
/// ```
pub fn gauss_chebyshev_nodes<T: LagRealTrait>(n: &usize, a: &T, b: &T) -> Vec<T> where i32: AsPrimitive<T> {
    (0..*n).map(|i| {
        let x = -Float::cos((Float::acos(-one::<T>())*(i as i32).as_())/((*n-1) as i32).as_());
        return rescale_range(a, b, &x);
    }).collect::<Vec<_>>()
}

/// Computes n linearly-spaced nodes on the interval `[a,b]`
/// 
/// # Example
/// 
/// ```
/// let (a,b,n) = (-1.0,1.0,3);
/// let x = linspace(&n,&a,&b); // should contain [-1.0,0.0,1.0]
/// ```
pub fn linspace<T:LagRealTrait>(n: &usize, a:&T, b: &T) -> Vec<T> {
    let stp = (*b-*a)/(T::from(*n-1).unwrap());
    (0..*n).map(|i| *a + T::from(i).unwrap()*stp).collect::<Vec<_>>()
}

/// Computes the weights for the barycentric formula: `w_j = \Pi_{m=0,m!=j}{1.0/(x[j]-x[m])}`
/// 
/// # Example
/// 
/// ```
/// let x = linspace(&3,&-1.0,&1.0);
/// let w = barycentric_weights(&x);
/// ```
pub fn barycentric_weights<T: LagRealTrait>(x: &Vec<T>) -> Vec<T> {
    x.iter().enumerate().map(|(j,&xj)| x.iter().enumerate().filter(|(m,_)| j != *m).map(|(_,&xm)| one::<T>()/(xj - xm)).product::<T>()).collect::<Vec<_>>()
}

/// Transpose of a Vec<Vec<_>> 
pub fn transpose_vec_of_vec<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>> {
    assert!(!v.is_empty());
    let len = v[0].len();
    let mut iters: Vec<_> = v.into_iter().map(|n| n.into_iter()).collect();
    (0..len).map(|_| {
        iters.iter_mut().map(|n| n.next().unwrap()).collect()
    }).collect()
}

#[doc(hidden)]
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

        let val = partial_sum(&xa, &x, j);
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

    #[test]
    pub fn accurate_barycentric_weights() {
        let x = vec![0.0,0.25,0.5,1.0];
        let w_ref = vec![
            1.0/((0.0-0.25)*(0.0-0.5)*(0.0-1.0)),
            1.0/((0.25-0.0)*(0.25-0.5)*(0.25-1.0)),
            1.0/((0.5-0.0)*(0.5-0.25)*(0.5-1.0)),
            1.0/((1.0-0.0)*(1.0-0.25)*(1.0-0.5))
        ];
        let w = barycentric_weights(&x);
        for i in 0..x.len() {
            assert!(f64::abs(w[i]-w_ref[i]) < 1e-14);
        }
    }
}