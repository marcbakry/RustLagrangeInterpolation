//! This module provides implementations of the univariate Lagrange interpolator for 
//! scalar (Lagrange1dInterpolator) and vector (Lagrange1dInterpolatorVec) real/complex
//! fields using the Rust standard library. It relies heavily on the `Vec` type.
//! 
//! # Cool features
//! 
//! All interpolators implement the Add/AddAssign, Sub/SubAssign, Mul/MulAssig, Div/DivAssign traits for
//! a scalar value or another interpolator, thus allowing function-like manipulations.
//! 
//!  Parallel evaluation of the
//! interpolator is available, based on the [`rayon`](rayon.rs) crate.
//! 
//! Computation of the derivatives of the interpolator are also available.
//! 
//! [`rayon`]: https://crates.io/crates/rayon
extern crate num_traits;
extern crate rayon;

pub mod lag1_utilities;

use num_traits::zero;
use num_traits::AsPrimitive;
use std::fmt::{Debug,Display,Formatter,Result};
use std::ops::{Div,DivAssign,Mul,MulAssign,Add,AddAssign,Sub,SubAssign};
use rayon::prelude::*;

use super::utilities::*;
use lag1_utilities::*;

/// The Lagrange1dInterpolator holds the data for the computation of the univariate
/// one-dimensional Lagrange interpolation.
#[derive(Debug,Clone)]
pub struct Lagrange1dInterpolator<T,U> {
    xa: Vec<T>,
    ya: Vec<U>,
    diff_order: usize
}

/// The Lagrange1dInterpolatorVec hold the data for the computation of the univariate
/// multidimensional Lagrange interpolation. It contains only a `Vec<Lagrange1dInterpolator>`.
#[derive(Debug,Clone)]
pub struct Lagrange1dInterpolatorVec<T,U> {
    lag1_interps: Vec<Lagrange1dInterpolator<T,U>>
}

impl<T,U> Lagrange1dInterpolatorVec<T,U> where 
T: LagRealTrait, i32: AsPrimitive<T>, U: LagComplexTrait + DivAssign<T> + MulAssign<T> {
    /// Returns a Lagrange1dInterpolatorVec for a serie of (xa,ya) interpolation data.
    /// 
    /// # Panics
    /// 
    /// This function will panic if the input sizes do not match, or if the individual
    /// interpolation data are ill-formed.
    /// 
    /// # Example
    /// 
    /// '''
    /// let xa = vec![vec![1.0,2.0,3.0],vec![1.5,2.5,3.5]];
    /// let ya = vec![vec![1.0,1.1,1.2],vec![1.2,1.1,1.0]];
    /// let i1d_vec = Lagrange1dInterpolatorVec::new(xa,ya);
    /// '''
    pub fn new(xa: Vec<Vec<T>>, ya: Vec<Vec<U>>) -> Lagrange1dInterpolatorVec<T,U> {
        if xa.len() != ya.len() {
            panic!("Error initializing the vector-field interpolator: inputs sizes do not match");
        }
        return Lagrange1dInterpolatorVec { 
            lag1_interps: ya.iter().zip(xa.iter()).map(|(y,x)| Lagrange1dInterpolator::new((*x).clone(), (*y).clone())).collect::<Vec<_>>() 
        };
    }

    /// Evaluates a Lagrange1dInterpolatorVec at some `x`. The output is a vector
    /// containing the value returned by each inner interpolator.
    /// 
    /// # Example
    /// 
    /// '''
    /// let i1d_vec = ...;
    /// let x = 0.0;
    /// let val = i1d_vec.eval(&x);
    /// '''
    pub fn eval(&self, x: &T) -> Vec<U> {
        return self.lag1_interps.iter().map(|interp| interp.eval(x)).collect::<Vec<U>>();
    }

    /// Evaluates a Lagrange1dInterpolatorVec for multiple `x`. The result is a
    /// vector of vectors containing the values returned by each inner interpolator.
    /// 
    /// # Example
    /// 
    /// '''
    /// let i1d_vec = ...;
    /// let x = [0.0,1.0];
    /// let val = i1d_vec.eval(&x);
    /// '''
    pub fn eval_vec(&self, x: &Vec<T>) -> Vec<Vec<U>> {
        // For each x-value, returns the value of all inner interpolators
        return x.iter().map(|x| self.eval(x)).collect::<Vec<_>>();
    }
    
    /// Parallel version of `.eval_vec()`.
    pub fn par_eval_vec(&self, x: &Vec<T>) -> Vec<Vec<U>>{
        return (*x).par_iter().map(|xx| self.eval(xx)).collect::<Vec<_>>();
    }

    /// Computes the first derivative of the current interpolator. The output is
    /// a new Lagrange1dInterpolatorVec interpolating the values of the derivative
    /// of the original inner interpolators.
    /// 
    /// # Example
    /// 
    /// '''
    /// let i1d_vec = ...;
    /// let i1d_vec_dx = i1d_vec.differentiate(); // new Lagrange1dInterpolatorVec
    /// '''
    pub fn differentiate(&self) -> Lagrange1dInterpolatorVec<T,U> {
        return Lagrange1dInterpolatorVec {
            lag1_interps: self.lag1_interps.iter().map(|interp| interp.differentiate()).collect::<Vec<_>>()
        };
    }

    /// Returns the inner Lagrange1dInterpolators in a vector.
    pub fn get_inner_interpolators(&self) -> Vec<Lagrange1dInterpolator<T, U>> {
        return self.lag1_interps.clone();
    }

    /// Returns the order of the inner interpolators in a vector.
    pub fn order(&self) -> Vec<usize> {
        return self.lag1_interps.iter().map(|interp| interp.order()).collect::<Vec<_>>();
    }

    /// Returns the length (<=> the number of nodes) of the inner interpolators in a vector.
    pub fn len(&self) -> Vec<usize> {
        return self.lag1_interps.iter().map(|interp| interp.len()).collect::<Vec<_>>();
    }

    /// Returns the dimension of the interpolated univariate vector field.
    pub fn dim(&self) -> usize {
        return self.lag1_interps.len();
    }
}

impl<T,U> Lagrange1dInterpolator<T,U> where 
T: LagRealTrait,
i32: AsPrimitive<T>,
U: LagComplexTrait + DivAssign<T> + MulAssign<T> {
    pub fn new(xa: Vec<T>, ya: Vec<U>) -> Lagrange1dInterpolator<T,U> {
        // check consistency
        if xa.len() != ya.len() {
            panic!("Invalid input: size of inputs do not match");
        }
        if xa.len() == 0 || ya.len() == 0 {
            panic!("Invalid input: 0-sized inputs");
        }
        check_duplicate(&xa);
        
        let indices = argsort(&xa);
        let xa = indices.iter().map(|&idx| xa[idx]).collect::<Vec<T>>();
        let ya = indices.iter().map(|&idx| ya[idx]).collect::<Vec<U>>();
        // 
        return Lagrange1dInterpolator{
            xa: xa,
            ya: ya,
            diff_order: 0
        };
    }

    pub fn order(&self) -> usize {
        self.xa.len()-1
    }

    pub fn len(&self) -> usize {
        self.xa.len()
    }

    pub fn diff_order(&self) -> usize {
        self.diff_order
    }

    pub fn get_interp_data(&self) -> (Vec<T>,Vec<U>) {
        (self.xa.clone(),self.ya.clone())
    }

    pub fn get_interp_data_ref(&self) -> (&Vec<T>,&Vec<U>) {
        (&(self.xa),&(self.ya))
    }

    pub fn eval(&self, x: &T) -> U {
        lag1_eval(&self.xa, &self.ya, x)
    }

    pub fn eval_vec(&self, x: &Vec<T>) -> Vec<U> {
        lag1_eval_vec(&self.xa, &self.ya, &x)
    }

    pub fn par_eval_vec(&self, x: &Vec<T>) -> Vec<U>{
        return (*x).par_iter().map(|xx| self.eval(xx)).collect::<Vec<U>>();
    }

    pub fn differentiate(&self) -> Lagrange1dInterpolator<T,U> {
        let (xa,ya) = self.get_interp_data();
        let n = self.len();
        let new_diff_order = self.diff_order()+1;

        if self.order() == 0 {
            return Lagrange1dInterpolator {
                xa: xa,
                ya: vec![zero::<U>(); n],
                diff_order: new_diff_order
            };
        } else {
            let xa_new = midpoints(&xa);
            let ya_new = lag1_eval_derivative_vec(&xa, &ya, &xa_new);

            return Lagrange1dInterpolator{
                xa: xa_new,
                ya: ya_new,
                diff_order: new_diff_order
            };
        }
    }

    pub fn differentiate_n_times(&self, n: usize) -> Lagrange1dInterpolator<T,U> {
        let mut dinterp = self.clone();
        for _ in 0..n {
            dinterp = dinterp.differentiate();
        }
        return dinterp;
    }
}

impl<T: LagRealTrait,U: LagComplexTrait> Display for Lagrange1dInterpolator<T,U> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f,"Lagrange 1d interpolator:\n- length = {}\n- differentiation order = {}",self.xa.len(),self.diff_order)
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> Add<U> for Lagrange1dInterpolator<T,U> {
    type Output = Lagrange1dInterpolator<T,U>;
    fn add(self, rhs: U) -> Self::Output {
        let mut new_self = self.clone();
        for e in &mut new_self.ya {
            *e = *e + rhs;
        }
        return new_self;
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> Add<Lagrange1dInterpolator<T,U>> for Lagrange1dInterpolator<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange1dInterpolator<T,U>;
    fn add(self, rhs: Lagrange1dInterpolator<T,U>) -> Self::Output {
        let (xa_lhs,ya_lhs) = self.get_interp_data();
        let (xa_rhs,ya_rhs) = rhs.get_interp_data();
        let is_same = xa_lhs.iter().zip(xa_rhs.iter()).any(|(&a,&b)| (a-b).abs() > T::from(1e-12).unwrap()) && (xa_lhs.len() == xa_rhs.len());
        if is_same { // if same interpolation nodes
            let new_ya = ya_lhs.iter().zip(ya_rhs.iter()).map(|(&a,&b)| a+b).collect::<Vec<_>>();
            return Lagrange1dInterpolator::new(xa_lhs, new_ya);
        } else { // if different, interpolation on for the interpolator with the more nodes
            let (new_xa,new_ya) = if self.len() > rhs.len() {
                let rhs_val = rhs.eval_vec(&xa_lhs);
                let new_ya = rhs_val.iter().zip(ya_lhs.iter()).map(|(&a,&b)| a+b).collect::<Vec<_>>();
                (xa_lhs,new_ya)
            } else {
                let lhs_val = self.eval_vec(&xa_rhs);
                let new_ya = lhs_val.iter().zip(ya_rhs.iter()).map(|(&a,&b)| a+b).collect::<Vec<_>>();
                (xa_rhs,new_ya)
            };
            return Lagrange1dInterpolator::new(new_xa,new_ya);
        }
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> AddAssign<U> for Lagrange1dInterpolator<T,U> {
    fn add_assign(&mut self, other: U) {
        for e in &mut self.ya {
            *e = *e + other;
        }
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> Sub<U> for Lagrange1dInterpolator<T,U> {
    type Output = Lagrange1dInterpolator<T,U>;
    fn sub(self, rhs: U) -> Self::Output {
        let mut new_self = self.clone();
        for e in &mut new_self.ya {
            *e = *e - rhs;
        }
        return new_self;
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> Sub<Lagrange1dInterpolator<T,U>> for Lagrange1dInterpolator<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange1dInterpolator<T,U>;
    fn sub(self, rhs: Lagrange1dInterpolator<T,U>) -> Self::Output {
        let (xa_lhs,ya_lhs) = self.get_interp_data();
        let (xa_rhs,ya_rhs) = rhs.get_interp_data();
        let is_same = xa_lhs.iter().zip(xa_rhs.iter()).any(|(&a,&b)| (a-b).abs() > T::from(1e-12).unwrap()) && (xa_lhs.len() == xa_rhs.len());
        if is_same { // if same interpolation nodes
            let new_ya = ya_lhs.iter().zip(ya_rhs.iter()).map(|(&a,&b)| a-b).collect::<Vec<_>>();
            return Lagrange1dInterpolator::new(xa_lhs, new_ya);
        } else { // if different, interpolation on for the interpolator with the more nodes
            let (new_xa,new_ya) = if self.len() > rhs.len() {
                let rhs_val = rhs.eval_vec(&xa_lhs);
                let new_ya = rhs_val.iter().zip(ya_lhs.iter()).map(|(&a,&b)| b-a).collect::<Vec<_>>();
                (xa_lhs,new_ya)
            } else {
                let lhs_val = self.eval_vec(&xa_rhs);
                let new_ya = lhs_val.iter().zip(ya_rhs.iter()).map(|(&a,&b)| a-b).collect::<Vec<_>>();
                (xa_rhs,new_ya)
            };
            return Lagrange1dInterpolator::new(new_xa,new_ya);
        }
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> SubAssign<U> for Lagrange1dInterpolator<T,U> {
    fn sub_assign(&mut self, other: U) {
        for e in &mut self.ya {
            *e = *e - other;
        }
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> Mul<U> for Lagrange1dInterpolator<T,U> {
    type Output = Lagrange1dInterpolator<T,U>;
    fn mul(self, rhs: U) -> Self::Output {
        let mut new_self = self.clone();
        for e in &mut new_self.ya {
            *e = (*e)*rhs;
        }
        return new_self;
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> Mul<Lagrange1dInterpolator<T,U>> for Lagrange1dInterpolator<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange1dInterpolator<T,U>;
    fn mul(self, rhs: Lagrange1dInterpolator<T,U>) -> Self::Output {
        let (xa_lhs,ya_lhs) = self.get_interp_data();
        let (xa_rhs,ya_rhs) = rhs.get_interp_data();
        let is_same = xa_lhs.iter().zip(xa_rhs.iter()).any(|(&a,&b)| (a-b).abs() > T::from(1e-12).unwrap()) && (xa_lhs.len() == xa_rhs.len());
        if is_same { // if same interpolation nodes
            let new_ya = ya_lhs.iter().zip(ya_rhs.iter()).map(|(&a,&b)| a*b).collect::<Vec<_>>();
            return Lagrange1dInterpolator::new(xa_lhs, new_ya);
        } else { // if different, interpolation on for the interpolator with the more nodes
            let (new_xa,new_ya) = if self.len() > rhs.len() {
                let rhs_val = rhs.eval_vec(&xa_lhs);
                let new_ya = rhs_val.iter().zip(ya_lhs.iter()).map(|(&a,&b)| a*b).collect::<Vec<_>>();
                (xa_lhs,new_ya)
            } else {
                let lhs_val = self.eval_vec(&xa_rhs);
                let new_ya = lhs_val.iter().zip(ya_rhs.iter()).map(|(&a,&b)| a*b).collect::<Vec<_>>();
                (xa_rhs,new_ya)
            };
            return Lagrange1dInterpolator::new(new_xa,new_ya);
        }
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> MulAssign<U> for Lagrange1dInterpolator<T,U> {
    fn mul_assign(&mut self, other: U) {
        for e in &mut self.ya {
            *e = (*e)*other;
        }
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> Div<U> for Lagrange1dInterpolator<T,U> {
    type Output = Lagrange1dInterpolator<T,U>;
    fn div(self, rhs: U) -> Self::Output {
        let mut new_self = self.clone();
        for e in &mut new_self.ya {
            *e = (*e)/rhs;
        }
        return new_self;
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> Div<Lagrange1dInterpolator<T,U>> for Lagrange1dInterpolator<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange1dInterpolator<T,U>;
    fn div(self, rhs: Lagrange1dInterpolator<T,U>) -> Self::Output {
        let (xa_lhs,ya_lhs) = self.get_interp_data();
        let (xa_rhs,ya_rhs) = rhs.get_interp_data();
        let is_same = xa_lhs.iter().zip(xa_rhs.iter()).any(|(&a,&b)| (a-b).abs() > T::from(1e-12).unwrap()) && (xa_lhs.len() == xa_rhs.len());
        if is_same { // if same interpolation nodes
            let new_ya = ya_lhs.iter().zip(ya_rhs.iter()).map(|(&a,&b)| a/b).collect::<Vec<_>>();
            return Lagrange1dInterpolator::new(xa_lhs, new_ya);
        } else { // if different, interpolation on for the interpolator with the more nodes
            let (new_xa,new_ya) = if self.len() > rhs.len() {
                let rhs_val = rhs.eval_vec(&xa_lhs);
                let new_ya = rhs_val.iter().zip(ya_lhs.iter()).map(|(&a,&b)| b/a).collect::<Vec<_>>();
                (xa_lhs,new_ya)
            } else {
                let lhs_val = self.eval_vec(&xa_rhs);
                let new_ya = lhs_val.iter().zip(ya_rhs.iter()).map(|(&a,&b)| a/b).collect::<Vec<_>>();
                (xa_rhs,new_ya)
            };
            return Lagrange1dInterpolator::new(new_xa,new_ya);
        }
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> DivAssign<U> for Lagrange1dInterpolator<T,U> {
    fn div_assign(&mut self, other: U) {
        for e in &mut self.ya {
            *e = (*e)/other;
        }
    }
}

// implementation of the basic operators for Lagrange1dInterpolatorVec
impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> Add<U> for Lagrange1dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange1dInterpolatorVec<T,U>;

    fn add(self, rhs: U) -> Self::Output {
        return Lagrange1dInterpolatorVec{
            lag1_interps: self.get_inner_interpolators().iter().map(|interp| interp.clone() + rhs).collect::<Vec<_>>()
        };
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> Add<Lagrange1dInterpolatorVec<T,U>> for Lagrange1dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange1dInterpolatorVec<T,U>;

    fn add(self, rhs: Lagrange1dInterpolatorVec<T,U>) -> Self::Output {
        let lhs = self.get_inner_interpolators();
        let rhs = rhs.get_inner_interpolators();
        return Lagrange1dInterpolatorVec{
            lag1_interps: lhs.iter().zip(rhs.iter()).map(|(i1,i2)| i1.clone()+i2.clone()).collect::<Vec<_>>()
        };
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> AddAssign<U> for Lagrange1dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    fn add_assign(&mut self, other: U) {
        for e in &mut self.lag1_interps {
            *e += other;
        }
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> Sub<U> for Lagrange1dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange1dInterpolatorVec<T,U>;

    fn sub(self, rhs: U) -> Self::Output {
        return Lagrange1dInterpolatorVec{
            lag1_interps: self.get_inner_interpolators().iter().map(|interp| interp.clone() - rhs).collect::<Vec<_>>()
        };
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> Sub<Lagrange1dInterpolatorVec<T,U>> for Lagrange1dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange1dInterpolatorVec<T,U>;

    fn sub(self, rhs: Lagrange1dInterpolatorVec<T,U>) -> Self::Output {
        let lhs = self.get_inner_interpolators();
        let rhs = rhs.get_inner_interpolators();
        return Lagrange1dInterpolatorVec{
            lag1_interps: lhs.iter().zip(rhs.iter()).map(|(i1,i2)| i1.clone()-i2.clone()).collect::<Vec<_>>()
        };
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> SubAssign<U> for Lagrange1dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    fn sub_assign(&mut self, other: U) {
        for e in &mut self.lag1_interps {
            *e -= other;
        }
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> Mul<U> for Lagrange1dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange1dInterpolatorVec<T,U>;

    fn mul(self, rhs: U) -> Self::Output {
        return Lagrange1dInterpolatorVec{
            lag1_interps: self.get_inner_interpolators().iter().map(|interp| interp.clone()*rhs).collect::<Vec<_>>()
        };
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> Mul<Lagrange1dInterpolatorVec<T,U>> for Lagrange1dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange1dInterpolatorVec<T,U>;

    fn mul(self, rhs: Lagrange1dInterpolatorVec<T,U>) -> Self::Output {
        let lhs = self.get_inner_interpolators();
        let rhs = rhs.get_inner_interpolators();
        return Lagrange1dInterpolatorVec{
            lag1_interps: lhs.iter().zip(rhs.iter()).map(|(i1,i2)| i1.clone()*i2.clone()).collect::<Vec<_>>()
        };
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> MulAssign<U> for Lagrange1dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    fn mul_assign(&mut self, other: U) {
        for e in &mut self.lag1_interps {
            *e *= other;
        }
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> Div<U> for Lagrange1dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange1dInterpolatorVec<T,U>;

    fn div(self, rhs: U) -> Self::Output {
        return Lagrange1dInterpolatorVec{
            lag1_interps: self.get_inner_interpolators().iter().map(|interp| interp.clone()/rhs).collect::<Vec<_>>()
        };
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> Div<Lagrange1dInterpolatorVec<T,U>> for Lagrange1dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange1dInterpolatorVec<T,U>;

    fn div(self, rhs: Lagrange1dInterpolatorVec<T,U>) -> Self::Output {
        let lhs = self.get_inner_interpolators();
        let rhs = rhs.get_inner_interpolators();
        return Lagrange1dInterpolatorVec{
            lag1_interps: lhs.iter().zip(rhs.iter()).map(|(i1,i2)| i1.clone()/i2.clone()).collect::<Vec<_>>()
        };
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> DivAssign<U> for Lagrange1dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    fn div_assign(&mut self, other: U) {
        for e in &mut self.lag1_interps {
            *e /= other;
        }
    }
}

// TESTS
#[cfg(test)]
pub mod lag1_tests;