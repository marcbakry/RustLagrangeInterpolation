//! This module provide implementations for the bivariate, **gridded**, Lagrange interpolator for
//! scalar (`Lagrange3dInterpolat`or) and vector (`Lagrange3dInterpolatorVec`) real/complex
//! fields using the Rust standard library. It relies heavily on the `Vec` type.
//! 
//! By **gridded**, we mean that the interpolation nodes should be dispatched over some
//! `x1a x x2a x x3a` grid with size `n1a x n2a x n3a` using the `x3a`- major convention: let `ya` be
//! the values at the interpolation nodes, the first `n3a` values correspond to `x1a[0],x2a[0]`, the next 
//! `n3a` to `x1a[0],x2a[1]`  and so on. Below, we show how we can interpolate some function `f(x1,x2,x3)` 
//! over the unit cube using a different number of nodes in each direction.
//! 
//! ```
//! use lagrange_interpolation::lagrange3d::*;
//! use lagrange_interpolation::utilities::*;
//! 
//! ...
//! 
//! let f = |x1:f64, x2: f64, x3: f64| f64::cos(2.0*std::f64::consts::PI*x1.powi(2))*x2.powf(1.5)-x3;
//! let (n1a,n2a,n3a) = (9,10,3);
//! let (a,b) = (0.0,1.0);
//! let (x1a,x2a,x3a) = (gauss_chebyshev_nodes(&n1a,&a,&b),gauss_chebyshev_nodes(&n2a,&a,&b),gauss_chebyshev_nodes(&n3a,&a,&b));
//! let ya = x1a.iter().map_flat(|x1| x2a.iter().map_flat(move |x2| x3a.iter().map(|x3| f(*x1,*x2,*x3)))).collect::<Vec<_>>();
//! let i3d = Lagrange2dInterpolator::new(x1a,x2a,x3a,ya);
//! let (x1,x2,x3) = (1.0/3.0,2.0/3.0,1.0/3.0);
//! let value = i3d.eval(&x1,&x2,&x3); // interpolation at a single value (x1,x2,x3)
//! ```
//! 
//! # Cool features
//! 
//! All interpolators implement the `Add/AddAssign`, `Sub/SubAssign`, `Mul/MulAssign`, `Div/DivAssign` traits for
//! a scalar value or another interpolator *of the same kind*, thus allowing 
//! function-like manipulations.
//! 
//! Parallel evaluation of the interpolator is available, based on the [rayon.rs](https://crates.io/crates/rayon) crate.
//! 
//! Computation of the partial derivatives of the interpolator are also available.
extern crate num_traits;

pub mod lag3_utilities;

use num_traits::{zero,AsPrimitive};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::fmt::{Debug,Display,Formatter,Result};
use std::ops::{Div,DivAssign,Mul,MulAssign,Add,AddAssign,Sub,SubAssign};

use lag3_utilities::*;

use super::utilities::*;

/// The `Lagrange3dInterpolator` structure holds the data for the computation of the
/// bivariate one-dimensional gridded Lagrange interpolation.
#[derive(Debug,Clone)]
pub struct Lagrange3dInterpolator<T,U> {
    x1a: Vec<T>,
    x2a: Vec<T>,
    x3a: Vec<T>,
    w1a: Vec<T>,
    w2a: Vec<T>,
    w3a: Vec<T>,
    ya: Vec<Vec<Vec<U>>>,
    diff1_order: usize,
    diff2_order: usize,
    diff3_order: usize
}

/// The `Lagrange3dInterpolatorVec` holds the data for the computation of the bivariate
/// multidimensional Lagrange interpolation. It contains only a `Vec<Lagrange3dInterpolator>`
/// but provides the same functionalities as `Lagrange3dInterpolator`.
#[derive(Debug,Clone)]
pub struct Lagrange3dInterpolatorVec<T,U> {
    lag3_interps: Vec<Lagrange3dInterpolator<T,U>>
}

impl<T,U> Lagrange3dInterpolatorVec<T,U> where
T: LagRealTrait, i32: AsPrimitive<T>, U: LagComplexTrait + LagBasicArithmetic<T> {
    /// Returns as `Lagrange3dInterpolatorVec` for a serie of `(x1a,x2a,x3a,ya)` interpolation data.
    /// 
    /// # Panics
    /// 
    /// This function will panic if the input sizes do not match, or if the individual 
    /// interpolation data are ill-formed (duplicated entries in `x1a` or `x2a`, for example).
    /// 
    /// # Example
    /// 
    /// ```
    /// let x1a = vec![vec![1.0,2.0,3.0],vec![1.5,2.5,3.5]];
    /// let x2a = vec![vec![1.0,2.0,3.0],vec![1.5,2.5,3.5]];
    /// let x3a = vec![vec![1.0,2.0,3.0],vec![1.5,2.5,3.5]];
    /// let ya = vec![vec![1.0;27],vec![2.0;27]];
    /// let i3d_vec = Lagrange2dInterpolatorVec::new(x1a,x2a,x3a,ya);
    /// ```
    pub fn new(x1a: Vec<Vec<T>>, x2a: Vec<Vec<T>>, x3a: Vec<Vec<T>>, ya: Vec<Vec<U>>) -> Lagrange3dInterpolatorVec<T,U> {
        if x1a.len() != x2a.len() || x1a.len() != x3a.len() || x1a.len() != ya.len() || x3a.len() != ya.len() {
            panic!("Error initializing the vector-field interpolator: inputs sizes do not match");
        }
        return Lagrange3dInterpolatorVec { 
            lag3_interps: ya.iter().zip(x1a.iter()).zip(x2a.iter()).zip(x3a.iter()).map(|(((y,x1),x2), x3)| Lagrange3dInterpolator::new((*x1).clone(), (*x2).clone(), (*x3).clone(), (*y).clone())).collect::<Vec<_>>()
         };
    }

    /// Evaluates a `Lagrange3dInterpolatorVec` at some `(x1,x2,x3)`. The output is a vector
    /// containing the value returned by each inner interpolator.
    pub fn eval(&self, x1: &T, x2: &T, x3: &T) -> Vec<U> {
        return self.lag3_interps.iter().map(|interp| interp.eval(x1, x2, x3)).collect::<Vec<_>>();
    }
    
    /// Evaluates `self` on a grid given by `x1`, `x2` and `x3` following the same ordering
    /// as the interpolation grid.
    pub fn eval_grid(&self, x1: &Vec<T>, x2: &Vec<T>, x3: &Vec<T>) -> Vec<Vec<U>>{
        return self.lag3_interps.iter().map(|interp| interp.eval_grid(x1, x2, x3)).collect::<Vec<_>>();
    }

    /// Evaluates `self` on a set of nodes whose coordinates are given in two separate vectors.
    /// The length of `x1` and `x2` and `x3` must match.
    pub fn eval_vec(&self, x1: &Vec<T>, x2: &Vec<T>, x3: &Vec<T>) -> Vec<Vec<U>> {
        return self.lag3_interps.iter().map(|interp| interp.eval_vec(x1, x2, x3)).collect::<Vec<_>>();
    }
    
    /// Evaluates `self` on a set of nodes whose coorinates are given in a vector 
    /// of size-two arrays.
    pub fn eval_arr(&self, x: &Vec<[T;3]>) -> Vec<Vec<U>> {
        return self.lag3_interps.iter().map(|interp| interp.eval_arr(x)).collect::<Vec<_>>();
    }

    /// Evaluates `self` on a set of nodes whose coorinates are given in a vector 
    /// of size-two tuples.
    pub fn eval_tup(&self, x: &Vec<(T,T,T)>) -> Vec<Vec<U>> {
        return self.lag3_interps.iter().map(|interp| interp.eval_tup(x)).collect::<Vec<_>>();
    }

    /// Parallel version of `self.eval_grid()`.
    pub fn par_eval_grid(&self, x1: &Vec<T>, x2: &Vec<T>, x3: &Vec<T>) -> Vec<Vec<U>>{
        return self.lag3_interps.iter().map(|interp| interp.par_eval_grid(x1, x2,x3)).collect::<Vec<_>>();
    }

    /// Parallel version of `self.eval_vec()`.
    pub fn par_eval_vec(&self, x1: &Vec<T>, x2: &Vec<T>, x3: &Vec<T>) -> Vec<Vec<U>> {
        return self.lag3_interps.iter().map(|interp| interp.par_eval_vec(x1, x2,x3)).collect::<Vec<_>>();
    }
    
    /// Parallel version of `self.eval_arr()`.
    pub fn par_eval_arr(&self, x: &Vec<[T;3]>) -> Vec<Vec<U>> {
        return self.lag3_interps.iter().map(|interp| interp.par_eval_arr(x)).collect::<Vec<_>>();
    }

    /// Parallel version of `self.eval_tup()`.
    pub fn par_eval_tup(&self, x: &Vec<(T,T,T)>) -> Vec<Vec<U>> {
        return self.lag3_interps.iter().map(|interp| interp.par_eval_tup(x)).collect::<Vec<_>>();
    }

    /// Computes the jacobian matrix of `self` as a vector of `Lagrange3dInterpolator`s. We recall that 
    /// the lines of the jacobian matrix hold the gradient of the associated component. Therefore, each
    /// entry of the output of this function holds a 3-array containing the components of the gradient.
    pub fn jacobian(&self) -> Vec<[Lagrange3dInterpolator<T,U>;3]> {
        self.lag3_interps.iter().map(|interp| [interp.differentiate_x1(),interp.differentiate_x2(),interp.differentiate_x3()]).collect::<Vec<_>>()
    }
    
    /// Get the inner interpolation data.
    pub fn get_inner_interpolators(&self) -> Vec<Lagrange3dInterpolator<T,U>> {
        return self.lag3_interps.clone();
    }

    /// Returns the interpolation order of each inner interpolator.
    pub fn order(&self) -> Vec<(usize,usize,usize)> {
        self.lag3_interps.iter().map(|interp| interp.order()).collect::<Vec<_>>()
    }
    
    /// Returns the number of interpolation nodes in each direction for each inner interpolator.
    pub fn len(&self) -> Vec<(usize,usize,usize)> {
        self.lag3_interps.iter().map(|interp| interp.len()).collect::<Vec<_>>()
    }

    /// Returns the dimension of the interpolated data.
    pub fn dim(&self) -> usize {
        return self.lag3_interps.len();
    }
}

impl<T,U> Lagrange3dInterpolator<T,U> where
T: LagRealTrait, i32: AsPrimitive<T>, U: LagComplexTrait + LagBasicArithmetic<T> {
    /// Returns a `Lagrange3dInterpolator` for some interpolation data `(x1a,x2a,x3a,ya)`.
    /// See the introduction of the module for the convention between the input data.
    /// 
    /// # Panics
    /// 
    /// This function will panic if the interpolation data is ill-formed.
    /// 
    /// # Example
    /// 
    /// ```
    /// let x1a = vec![1.0,2.0,3.0];
    /// let x2a = vec![0.0,2.0];
    /// let x3a = vec![0.0,2.0,4.0,6.0];
    /// let ya = vec![1.0;3*2*4]; // interpolate a constant function
    /// let i3d = Lagrange2dInterpolator::new(x1a,x2,x3a,ya);
    /// ```
    pub fn new(x1a: Vec<T>, x2a: Vec<T>, x3a: Vec<T>, ya: Vec<U>) -> Lagrange3dInterpolator<T,U> {
        // 
        if x1a.len()*x2a.len()*x3a.len() != ya.len() {
            panic!("Invalid input: size of inputs do not match")
        }
        if x1a.len() == 0 || x2a.len()  == 0 || x3a.len() == 0 || ya.len() == 0 {
            panic!("Invalid input: 0-sized inputs");
        }
        check_duplicate(&x1a);
        check_duplicate(&x2a);
        check_duplicate(&x3a);

        let idx1a = argsort(&x1a);
        let idx2a = argsort(&x2a);
        let idx3a = argsort(&x3a);

        let x1a = idx1a.iter().map(|&i| x1a[i]).collect::<Vec<T>>();
        let x2a = idx2a.iter().map(|&i| x2a[i]).collect::<Vec<T>>();
        let x3a = idx3a.iter().map(|&i| x3a[i]).collect::<Vec<T>>();

        let w1a = barycentric_weights(&x1a);
        let w2a = barycentric_weights(&x2a);
        let w3a = barycentric_weights(&x3a);

        let mut ya_sorted = Vec::with_capacity(x1a.len());
        for i1a in 0..x1a.len() {
            let mut tmp = Vec::with_capacity(x2a.len());
            for i2a in 0..x2a.len() {
                let beg = i1a*x2a.len()*x3a.len() + i2a*x3a.len();
                tmp.push(idx3a.iter().map(|&idx3| ya[idx3+beg]).collect::<Vec<U>>());
            }
            // sort tmp
            ya_sorted.push(idx2a.iter().map(|&idx2| tmp[idx2].clone()).collect::<Vec<_>>());
        }
        let ya = idx1a.into_iter().map(|idx1| ya_sorted
        [idx1].clone()).collect::<Vec<_>>();
        // 
        return Lagrange3dInterpolator{
            x1a: x1a,
            x2a: x2a,
            x3a: x3a,
            w1a: w1a,
            w2a: w2a,
            w3a: w3a,
            ya: ya,
            diff1_order: 0,
            diff2_order: 0,
            diff3_order: 0
        };
    }

    /// Returns the order of the interpolating polynomial in each direction
    pub fn order(&self) -> (usize,usize,usize) {
        (self.x1a.len()-1,self.x2a.len()-1,self.x3a.len()-1)
    }
    
    /// Returns the number of interpolation nodes in each direction.
    pub fn len(&self) -> (usize,usize,usize) {
        (self.x1a.len(),self.x2a.len(),self.x3a.len())
    }
    
    /// Returns the differentiation order with respect to each of the variables.
    pub fn diff_order(&self) -> (usize,usize,usize) {
        (self.diff1_order,self.diff2_order,self.diff3_order)
    }

    /// Get a copy of the underlying interpolation data
    pub fn get_interp_data(&self) -> (Vec<T>,Vec<T>,Vec<T>,Vec<Vec<Vec<U>>>) {
        (self.x1a.clone(),self.x2a.clone(),self.x3a.clone(),self.ya.clone())
    }

    /// Get a reference on the interpolation data
    pub fn get_interp_data_ref(&self) -> (&Vec<T>,&Vec<T>,&Vec<T>,&Vec<Vec<Vec<U>>>) {
        (&(self.x1a),&(self.x2a),&(self.x3a),&(self.ya))
    }

    /// Evaluates the interpolator at some `(x1,x2,x3)`.
    pub fn eval(&self, x1: &T, x2: &T, x3: &T) -> U {
        // lag3_eval(&self.x1a, &self.x2a, &self.x3a, &self.ya, x1, x2, x3)
        lag3_eval_barycentric(&self.x1a, &self.x2a, &self.x3a, &self.w1a, &self.w2a, &self.w3a, &self.ya, x1, x2, x3)
    }

    /// Evaluates `self` on a grid given by `x1`, `x2` and `x3` following the same ordering
    /// as the interpolation grid.
    pub fn eval_grid(&self, x1: &Vec<T>, x2: &Vec<T>, x3: &Vec<T>) -> Vec<U> {
        // lag3_eval_grid(&self.x1a, &self.x2a, &self.x3a, &self.ya, x1, x2,x3)
        lag3_eval_grid_barycentric(&self.x1a, &self.x2a, &self.x3a, &self.w1a, &self.w2a, &self.w3a, &self.ya, x1, x2,x3)
    }

    /// Evaluates `self` on a set of nodes whose coordinates are given in two separate vectors.
    /// The length of `x1`, `x2` and `x3` must match.
    pub fn eval_vec(&self, x1: &Vec<T>, x2: &Vec<T>, x3: &Vec<T>) -> Vec<U> {
        // lag3_eval_vec(&self.x1a, &self.x2a, &self.x3a, &self.ya, x1, x2, x3)
        lag3_eval_vec_barycentric(&self.x1a, &self.x2a, &self.x3a, &self.w1a, &self.w2a, &self.w3a, &self.ya, x1, x2, x3)
    }
    
    /// Evaluates `self` on a set of nodes whose coorinates are given in a vector 
    /// of size-two arrays.
    pub fn eval_arr(&self, x: &Vec<[T;3]>) -> Vec<U> {
        // x.iter().map(|e| lag3_eval(&self.x1a, &self.x2a, &self.x3a, &self.ya, &e[0], &e[1],&e[2])).collect::<Vec<_>>()
        x.iter().map(|e| lag3_eval_barycentric(&self.x1a, &self.x2a, &self.x3a, &self.w1a, &self.w2a, &self.w3a, &self.ya, &e[0], &e[1],&e[2])).collect::<Vec<_>>()
    }

    /// Evaluates `self` on a set of nodes whose coorinates are given in a vector 
    /// of size-two tuples.
    pub fn eval_tup(&self, x: &Vec<(T,T,T)>) -> Vec<U> {
        // x.iter().map(|e| lag3_eval(&self.x1a, &self.x2a, &self.x3a, &self.ya, &e.0, &e.1, &e.2)).collect::<Vec<_>>()
        x.iter().map(|e| lag3_eval_barycentric(&self.x1a, &self.x2a, &self.x3a, &self.w1a, &self.w2a, &self.w3a, &self.ya, &e.0, &e.1, &e.2)).collect::<Vec<_>>()
    }

    /// Parallel version of `self.eval_grid()`.
    pub fn par_eval_grid(&self, x1: &Vec<T>, x2: &Vec<T>, x3: &Vec<T>) -> Vec<U> {
        (*x1).par_iter().flat_map_iter(|xx1| (*x2).iter().flat_map(|xx2| (*x3).iter().map(|xx3| self.eval(xx1, xx2, xx3)))).collect::<Vec<_>>()
    }

    /// Parallel version of `self.eval_vec()`.
    pub fn par_eval_vec(&self, x1: &Vec<T>, x2: &Vec<T>, x3: &Vec<T>) -> Vec<U> {
        (*x1).par_iter().zip_eq((*x2).par_iter()).zip_eq((*x3).par_iter()).map(|((xx1,xx2),xx3)| self.eval(xx1, xx2, xx3)).collect::<Vec<_>>()
    }

    /// Parallel version of `self.eval_arr()`.
    pub fn par_eval_arr(&self, x: &Vec<[T;3]>) -> Vec<U> {
        (*x).par_iter().map(|&xx| self.eval(&xx[0], &xx[1], &xx[2])).collect::<Vec<U>>()
    }

    /// Parallel version of `self.eval_tup()`.
    pub fn par_eval_tup(&self, x: &Vec<(T,T,T)>) -> Vec<U> {
        (*x).par_iter().map(|&(x1,x2,x3)| self.eval(&x1,&x2,&x3)).collect::<Vec<_>>()
    }

    /// Returns the partial derivative with respect to `x1` of `self` as a new `Lagrange3dInterpolator` 
    /// on `self.len().0-1` nodes. If the length of the new interpolator falls 
    /// to 0, it returns instead a `Lagrange3dInterpolator` with a single node
    /// and the value 0.
    pub fn differentiate_x1(&self) -> Lagrange3dInterpolator<T, U> {
        // 
        let (x1a,x2a,x3a,ya) = self.get_interp_data();
        let new_diff1_order = self.diff1_order + 1;
        // 
        if self.x1a.len()-1 == 0 {
            let n = x1a.len()*x2a.len()*x3a.len();
            let mut zero_interp = Lagrange3dInterpolator::new(x1a, x2a, x3a, vec![zero::<U>(); n]);
            zero_interp.diff1_order = new_diff1_order;
            zero_interp.diff2_order = self.diff2_order;
            zero_interp.diff3_order = self.diff3_order;
            return zero_interp;
        } else {
            let x1a_new = midpoints(&x1a);
            let w1a = barycentric_weights(&x1a);
            let w2a = barycentric_weights(&x2a);
            let w3a = barycentric_weights(&x3a);
            let ya_new = lag3_diff1_grid_barycentric(&x1a, &x2a, &x3a, &w1a, &w2a, &w3a, &ya, &x1a_new, &x2a, &x3a);

            let mut output = Lagrange3dInterpolator::new(x1a_new,x2a,x3a,ya_new);
            output.diff1_order = new_diff1_order;
            output.diff2_order = self.diff2_order;
            output.diff3_order = self.diff3_order;
            return output;
        }
    }

    /// Returns the partial derivative with respect to `x2` of `self` as a new `Lagrange3dInterpolator` 
    /// on `self.len().1-1` nodes. If the length of the new interpolator falls 
    /// to 0, it returns instead a `Lagrange3dInterpolator` with a single node
    /// and the value 0.
    pub fn differentiate_x2(&self) -> Lagrange3dInterpolator<T, U> {
        // 
        let (x1a,x2a,x3a,ya) = self.get_interp_data();
        let new_diff2_order = self.diff2_order + 1;
        // 
        if self.x2a.len()-1 == 0 {
            let n = x1a.len()*x2a.len()*x3a.len();
            let mut zero_interp = Lagrange3dInterpolator::new(x1a, x2a, x3a, vec![zero::<U>(); n]);
            zero_interp.diff1_order = self.diff1_order;
            zero_interp.diff2_order = new_diff2_order;
            zero_interp.diff3_order = self.diff3_order;
            return zero_interp;
        } else {
            let x2a_new = midpoints(&x2a);
            let w1a = barycentric_weights(&x1a);
            let w2a = barycentric_weights(&x2a);
            let w3a = barycentric_weights(&x3a);
            let ya_new = lag3_diff2_grid_barycentric(&x1a, &x2a, &x3a, &w1a, &w2a, &w3a, &ya, &x1a, &x2a_new, &x3a);

            let mut output = Lagrange3dInterpolator::new(x1a,x2a_new,x3a,ya_new);
            output.diff1_order = self.diff1_order;
            output.diff2_order = new_diff2_order;
            output.diff3_order = self.diff3_order;
            return output;
        }
    }

    /// Returns the partial derivative with respect to `x3` of `self` as a new `Lagrange3dInterpolator` 
    /// on `self.len().2-1` nodes. If the length of the new interpolator falls 
    /// to 0, it returns instead a `Lagrange3dInterpolator` with a single node
    /// and the value 0.
    pub fn differentiate_x3(&self) -> Lagrange3dInterpolator<T, U> {
        // 
        let (x1a,x2a,x3a,ya) = self.get_interp_data();
        let new_diff3_order = self.diff3_order + 1;
        // 
        if self.x3a.len()-1 == 0 {
            let n = x1a.len()*x2a.len()*x3a.len();
            let mut zero_interp = Lagrange3dInterpolator::new(x1a, x2a, x3a, vec![zero::<U>(); n]);
            zero_interp.diff1_order = self.diff1_order;
            zero_interp.diff2_order = self.diff2_order;
            zero_interp.diff3_order = new_diff3_order;
            return zero_interp;
        } else {
            let x3a_new = midpoints(&x3a);
            let w1a = barycentric_weights(&x1a);
            let w2a = barycentric_weights(&x2a);
            let w3a = barycentric_weights(&x3a);
            let ya_new = lag3_diff3_grid_barycentric(&x1a, &x2a, &x3a, &w1a, &w2a, &w3a, &ya, &x1a, &x2a, &x3a_new);

            let mut output = Lagrange3dInterpolator::new(x1a,x2a,x3a_new,ya_new);
            output.diff1_order = self.diff1_order;
            output.diff2_order = self.diff2_order;
            output.diff3_order = new_diff3_order;
            return output;
        }
    }

    /// Computes the gradient of the interpolator and returns it as a
    /// size-3 array.
    pub fn gradient(&self) -> [Lagrange3dInterpolator<T,U>;3] {
        return [self.differentiate_x1(),self.differentiate_x2(),self.differentiate_x3()] ;
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T>> Display for Lagrange3dInterpolator<T,U> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f,"Lagrange 2d gridded interpolator:\n- length = {} x {} x {}\n- differentiation order = ({},{},{})",self.x1a.len(),self.x2a.len(), self.x3a.len(),self.diff1_order,self.diff2_order,self.diff3_order)
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T>> Add<U> for Lagrange3dInterpolator<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange3dInterpolator<T,U>;
    fn add(self, rhs: U) -> Self::Output {
        let (x1a,x2a,x3a,ya) = self.get_interp_data();
        let new_ya = ya.iter().flat_map(|y| y.iter().flat_map(|yy| yy)).map(|&e| e+rhs).collect::<Vec<_>>();
        return Lagrange3dInterpolator::new(x1a, x2a, x3a, new_ya);
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T>> Add<Lagrange3dInterpolator<T,U>> for Lagrange3dInterpolator<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange3dInterpolator<T,U>; 
    fn add(self, rhs: Lagrange3dInterpolator<T,U>) -> Self::Output {
        let (x1a_lhs,x2a_lhs,x3a_lhs,ya_lhs) = self.get_interp_data();
        let (x1a_rhs,x2a_rhs,x3a_rhs,ya_rhs) = rhs.get_interp_data();
        
        let is_same_x1a = x1a_lhs.iter().zip(x1a_rhs.iter()).any(|(&a,&b)| (a-b).abs() > T::from(1e-12).unwrap()) && (x1a_lhs.len() == x1a_rhs.len());
        let is_same_x2a = x2a_lhs.iter().zip(x2a_rhs.iter()).any(|(&a,&b)| (a-b).abs() > T::from(1e-12).unwrap()) && (x2a_lhs.len() == x2a_rhs.len());
        let is_same_x3a = x3a_lhs.iter().zip(x3a_rhs.iter()).any(|(&a,&b)| (a-b).abs() > T::from(1e-12).unwrap()) && (x3a_lhs.len() == x3a_rhs.len());

        // flatten y
        let ya_lhs = ya_lhs.iter().flat_map(|e| (*e).iter()).flat_map(|e| (*e).clone()).collect::<Vec<_>>();
        let ya_rhs = ya_rhs.iter().flat_map(|e| (*e).iter()).flat_map(|e| (*e).clone()).collect::<Vec<_>>();

        let (x1a_new,x2a_new, x3a_new,ya_new) = if is_same_x1a && is_same_x2a && is_same_x3a {
            (x1a_lhs,x2a_lhs,x3a_lhs,ya_lhs.iter().zip(ya_rhs.iter()).map(|(&a,&b)| a+b).collect::<Vec<_>>())
        } else {
            let x1a_new = if x1a_lhs.len() > x1a_rhs.len() {
                x1a_lhs
            } else {
                x1a_rhs
            };
            let x2a_new = if x2a_lhs.len() > x2a_rhs.len() {
                x2a_lhs
            } else {
                x2a_rhs
            };
            let x3a_new = if x3a_lhs.len() > x3a_rhs.len() {
                x3a_lhs
            } else {
                x3a_rhs
            };
            let ya_lhs = self.eval_grid(&x1a_new, &x2a_new,&x3a_new);
            let ya_rhs = rhs.eval_grid(&x1a_new, &x2a_new,&x3a_new);
            let ya_new = ya_lhs.iter().zip(ya_rhs.iter()).map(|(&a,&b)| a+b).collect::<Vec<_>>();
            (x1a_new,x2a_new,x3a_new,ya_new)
        };
        return Lagrange3dInterpolator::new(x1a_new,x2a_new,x3a_new,ya_new);
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T>> AddAssign<U> for Lagrange3dInterpolator<T,U> where i32: AsPrimitive<T> {
    fn add_assign(&mut self, rhs: U) {
        for y in &mut self.ya {
            for yy in y {
                for yyy in yy {
                    *yyy = *yyy + rhs;
                }
            }
        }
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T>> Sub<U> for Lagrange3dInterpolator<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange3dInterpolator<T,U>;
    fn sub(self, rhs: U) -> Self::Output {
        let (x1a,x2a,x3a,ya) = self.get_interp_data();
        let new_ya = ya.iter().flat_map(|y| y.iter().flat_map(|yy| yy)).map(|&e| e-rhs).collect::<Vec<_>>();
        return Lagrange3dInterpolator::new(x1a, x2a, x3a, new_ya);
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T>> Sub<Lagrange3dInterpolator<T,U>> for Lagrange3dInterpolator<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange3dInterpolator<T,U>; 
    fn sub(self, rhs: Lagrange3dInterpolator<T,U>) -> Self::Output {
        let (x1a_lhs,x2a_lhs,x3a_lhs,ya_lhs) = self.get_interp_data();
        let (x1a_rhs,x2a_rhs,x3a_rhs,ya_rhs) = rhs.get_interp_data();
        
        let is_same_x1a = x1a_lhs.iter().zip(x1a_rhs.iter()).any(|(&a,&b)| (a-b).abs() > T::from(1e-12).unwrap()) && (x1a_lhs.len() == x1a_rhs.len());
        let is_same_x2a = x2a_lhs.iter().zip(x2a_rhs.iter()).any(|(&a,&b)| (a-b).abs() > T::from(1e-12).unwrap()) && (x2a_lhs.len() == x2a_rhs.len());
        let is_same_x3a = x3a_lhs.iter().zip(x3a_rhs.iter()).any(|(&a,&b)| (a-b).abs() > T::from(1e-12).unwrap()) && (x3a_lhs.len() == x3a_rhs.len());

        // flatten y
        let ya_lhs = ya_lhs.iter().flat_map(|e| (*e).iter()).flat_map(|e| (*e).clone()).collect::<Vec<_>>();
        let ya_rhs = ya_rhs.iter().flat_map(|e| (*e).iter()).flat_map(|e| (*e).clone()).collect::<Vec<_>>();

        let (x1a_new,x2a_new, x3a_new,ya_new) = if is_same_x1a && is_same_x2a && is_same_x3a {
            (x1a_lhs,x2a_lhs,x3a_lhs,ya_lhs.iter().zip(ya_rhs.iter()).map(|(&a,&b)| a-b).collect::<Vec<_>>())
        } else {
            let x1a_new = if x1a_lhs.len() > x1a_rhs.len() {
                x1a_lhs
            } else {
                x1a_rhs
            };
            let x2a_new = if x2a_lhs.len() > x2a_rhs.len() {
                x2a_lhs
            } else {
                x2a_rhs
            };
            let x3a_new = if x3a_lhs.len() > x3a_rhs.len() {
                x3a_lhs
            } else {
                x3a_rhs
            };
            let ya_lhs = self.eval_grid(&x1a_new, &x2a_new,&x3a_new);
            let ya_rhs = rhs.eval_grid(&x1a_new, &x2a_new,&x3a_new);
            let ya_new = ya_lhs.iter().zip(ya_rhs.iter()).map(|(&a,&b)| a-b).collect::<Vec<_>>();
            (x1a_new,x2a_new,x3a_new,ya_new)
        };
        return Lagrange3dInterpolator::new(x1a_new,x2a_new,x3a_new,ya_new);
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T>> SubAssign<U> for Lagrange3dInterpolator<T,U> where i32: AsPrimitive<T> {
    fn sub_assign(&mut self, rhs: U) {
        for y in &mut self.ya {
            for yy in y {
                for yyy in yy {
                    *yyy = *yyy - rhs;
                }
            }
        }
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T>> Mul<U> for Lagrange3dInterpolator<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange3dInterpolator<T,U>;
    fn mul(self, rhs: U) -> Self::Output {
        let (x1a,x2a,x3a,ya) = self.get_interp_data();
        let new_ya = ya.iter().flat_map(|y| y.iter().flat_map(|yy| yy)).map(|&e| e*rhs).collect::<Vec<_>>();
        return Lagrange3dInterpolator::new(x1a, x2a, x3a, new_ya);
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T>> Mul<Lagrange3dInterpolator<T,U>> for Lagrange3dInterpolator<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange3dInterpolator<T,U>; 
    fn mul(self, rhs: Lagrange3dInterpolator<T,U>) -> Self::Output {
        let (x1a_lhs,x2a_lhs,x3a_lhs,ya_lhs) = self.get_interp_data();
        let (x1a_rhs,x2a_rhs,x3a_rhs,ya_rhs) = rhs.get_interp_data();
        
        let is_same_x1a = x1a_lhs.iter().zip(x1a_rhs.iter()).any(|(&a,&b)| (a-b).abs() > T::from(1e-12).unwrap()) && (x1a_lhs.len() == x1a_rhs.len());
        let is_same_x2a = x2a_lhs.iter().zip(x2a_rhs.iter()).any(|(&a,&b)| (a-b).abs() > T::from(1e-12).unwrap()) && (x2a_lhs.len() == x2a_rhs.len());
        let is_same_x3a = x3a_lhs.iter().zip(x3a_rhs.iter()).any(|(&a,&b)| (a-b).abs() > T::from(1e-12).unwrap()) && (x3a_lhs.len() == x3a_rhs.len());

        // flatten y
        let ya_lhs = ya_lhs.iter().flat_map(|e| (*e).iter()).flat_map(|e| (*e).clone()).collect::<Vec<_>>();
        let ya_rhs = ya_rhs.iter().flat_map(|e| (*e).iter()).flat_map(|e| (*e).clone()).collect::<Vec<_>>();

        let (x1a_new,x2a_new, x3a_new,ya_new) = if is_same_x1a && is_same_x2a && is_same_x3a {
            (x1a_lhs,x2a_lhs,x3a_lhs,ya_lhs.iter().zip(ya_rhs.iter()).map(|(&a,&b)| a*b).collect::<Vec<_>>())
        } else {
            let x1a_new = if x1a_lhs.len() > x1a_rhs.len() {
                x1a_lhs
            } else {
                x1a_rhs
            };
            let x2a_new = if x2a_lhs.len() > x2a_rhs.len() {
                x2a_lhs
            } else {
                x2a_rhs
            };
            let x3a_new = if x3a_lhs.len() > x3a_rhs.len() {
                x3a_lhs
            } else {
                x3a_rhs
            };
            let ya_lhs = self.eval_grid(&x1a_new, &x2a_new,&x3a_new);
            let ya_rhs = rhs.eval_grid(&x1a_new, &x2a_new,&x3a_new);
            let ya_new = ya_lhs.iter().zip(ya_rhs.iter()).map(|(&a,&b)| a*b).collect::<Vec<_>>();
            (x1a_new,x2a_new,x3a_new,ya_new)
        };
        return Lagrange3dInterpolator::new(x1a_new,x2a_new,x3a_new,ya_new);
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T>> MulAssign<U> for Lagrange3dInterpolator<T,U> where i32: AsPrimitive<T> {
    fn mul_assign(&mut self, rhs: U) {
        for y in &mut self.ya {
            for yy in y {
                for yyy in yy {
                    *yyy = *yyy*rhs;
                }
            }
        }
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T>> Div<U> for Lagrange3dInterpolator<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange3dInterpolator<T,U>;
    fn div(self, rhs: U) -> Self::Output {
        let (x1a,x2a,x3a,ya) = self.get_interp_data();
        let new_ya = ya.iter().flat_map(|y| y.iter().flat_map(|yy| yy)).map(|&e| e/rhs).collect::<Vec<_>>();
        return Lagrange3dInterpolator::new(x1a, x2a, x3a, new_ya);
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T>> Div<Lagrange3dInterpolator<T,U>> for Lagrange3dInterpolator<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange3dInterpolator<T,U>; 
    fn div(self, rhs: Lagrange3dInterpolator<T,U>) -> Self::Output {
        let (x1a_lhs,x2a_lhs,x3a_lhs,ya_lhs) = self.get_interp_data();
        let (x1a_rhs,x2a_rhs,x3a_rhs,ya_rhs) = rhs.get_interp_data();
        
        let is_same_x1a = x1a_lhs.iter().zip(x1a_rhs.iter()).any(|(&a,&b)| (a-b).abs() > T::from(1e-12).unwrap()) && (x1a_lhs.len() == x1a_rhs.len());
        let is_same_x2a = x2a_lhs.iter().zip(x2a_rhs.iter()).any(|(&a,&b)| (a-b).abs() > T::from(1e-12).unwrap()) && (x2a_lhs.len() == x2a_rhs.len());
        let is_same_x3a = x3a_lhs.iter().zip(x3a_rhs.iter()).any(|(&a,&b)| (a-b).abs() > T::from(1e-12).unwrap()) && (x3a_lhs.len() == x3a_rhs.len());

        // flatten y
        let ya_lhs = ya_lhs.iter().flat_map(|e| (*e).iter()).flat_map(|e| (*e).clone()).collect::<Vec<_>>();
        let ya_rhs = ya_rhs.iter().flat_map(|e| (*e).iter()).flat_map(|e| (*e).clone()).collect::<Vec<_>>();

        let (x1a_new,x2a_new, x3a_new,ya_new) = if is_same_x1a && is_same_x2a && is_same_x3a {
            (x1a_lhs,x2a_lhs,x3a_lhs,ya_lhs.iter().zip(ya_rhs.iter()).map(|(&a,&b)| a/b).collect::<Vec<_>>())
        } else {
            let x1a_new = if x1a_lhs.len() > x1a_rhs.len() {
                x1a_lhs
            } else {
                x1a_rhs
            };
            let x2a_new = if x2a_lhs.len() > x2a_rhs.len() {
                x2a_lhs
            } else {
                x2a_rhs
            };
            let x3a_new = if x3a_lhs.len() > x3a_rhs.len() {
                x3a_lhs
            } else {
                x3a_rhs
            };
            let ya_lhs = self.eval_grid(&x1a_new, &x2a_new,&x3a_new);
            let ya_rhs = rhs.eval_grid(&x1a_new, &x2a_new,&x3a_new);
            let ya_new = ya_lhs.iter().zip(ya_rhs.iter()).map(|(&a,&b)| a/b).collect::<Vec<_>>();
            (x1a_new,x2a_new,x3a_new,ya_new)
        };
        return Lagrange3dInterpolator::new(x1a_new,x2a_new,x3a_new,ya_new);
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T>> DivAssign<U> for Lagrange3dInterpolator<T,U> where i32: AsPrimitive<T> {
    fn div_assign(&mut self, rhs: U) {
        for y in &mut self.ya {
            for yy in y {
                for yyy in yy {
                    *yyy = *yyy/rhs;
                }
            }
        }
    }
}

// implementation of the basic operators for Lagrange3dInterpolatorVec
impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T>> Add<U> for Lagrange3dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange3dInterpolatorVec<T,U>;

    fn add(self, rhs: U) -> Self::Output {
        return Lagrange3dInterpolatorVec{
            lag3_interps: self.get_inner_interpolators().iter().map(|interp| interp.clone() + rhs).collect::<Vec<_>>()
        };
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T>> Add<Lagrange3dInterpolatorVec<T,U>> for Lagrange3dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange3dInterpolatorVec<T,U>;

    fn add(self, rhs: Lagrange3dInterpolatorVec<T,U>) -> Self::Output {
        let lhs = self.get_inner_interpolators();
        let rhs = rhs.get_inner_interpolators();
        return Lagrange3dInterpolatorVec{
            lag3_interps: lhs.iter().zip(rhs.iter()).map(|(i1,i2)| i1.clone()+i2.clone()).collect::<Vec<_>>()
        };
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T>> AddAssign<U> for Lagrange3dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    fn add_assign(&mut self, other: U) {
        for e in &mut self.lag3_interps {
            *e += other;
        }
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T>> Sub<U> for Lagrange3dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange3dInterpolatorVec<T,U>;

    fn sub(self, rhs: U) -> Self::Output {
        return Lagrange3dInterpolatorVec{
            lag3_interps: self.get_inner_interpolators().iter().map(|interp| interp.clone() - rhs).collect::<Vec<_>>()
        };
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T>> Sub<Lagrange3dInterpolatorVec<T,U>> for Lagrange3dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange3dInterpolatorVec<T,U>;

    fn sub(self, rhs: Lagrange3dInterpolatorVec<T,U>) -> Self::Output {
        let lhs = self.get_inner_interpolators();
        let rhs = rhs.get_inner_interpolators();
        return Lagrange3dInterpolatorVec{
            lag3_interps: lhs.iter().zip(rhs.iter()).map(|(i1,i2)| i1.clone()-i2.clone()).collect::<Vec<_>>()
        };
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T>> SubAssign<U> for Lagrange3dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    fn sub_assign(&mut self, other: U) {
        for e in &mut self.lag3_interps {
            *e -= other;
        }
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T>> Mul<U> for Lagrange3dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange3dInterpolatorVec<T,U>;

    fn mul(self, rhs: U) -> Self::Output {
        return Lagrange3dInterpolatorVec{
            lag3_interps: self.get_inner_interpolators().iter().map(|interp| interp.clone()*rhs).collect::<Vec<_>>()
        };
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T>> Mul<Lagrange3dInterpolatorVec<T,U>> for Lagrange3dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange3dInterpolatorVec<T,U>;

    fn mul(self, rhs: Lagrange3dInterpolatorVec<T,U>) -> Self::Output {
        let lhs = self.get_inner_interpolators();
        let rhs = rhs.get_inner_interpolators();
        return Lagrange3dInterpolatorVec{
            lag3_interps: lhs.iter().zip(rhs.iter()).map(|(i1,i2)| i1.clone()*i2.clone()).collect::<Vec<_>>()
        };
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T>> MulAssign<U> for Lagrange3dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    fn mul_assign(&mut self, other: U) {
        for e in &mut self.lag3_interps {
            *e *= other;
        }
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T>> Div<U> for Lagrange3dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange3dInterpolatorVec<T,U>;

    fn div(self, rhs: U) -> Self::Output {
        return Lagrange3dInterpolatorVec{
            lag3_interps: self.get_inner_interpolators().iter().map(|interp| interp.clone()/rhs).collect::<Vec<_>>()
        };
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T>> Div<Lagrange3dInterpolatorVec<T,U>> for Lagrange3dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange3dInterpolatorVec<T,U>;

    fn div(self, rhs: Lagrange3dInterpolatorVec<T,U>) -> Self::Output {
        let lhs = self.get_inner_interpolators();
        let rhs = rhs.get_inner_interpolators();
        return Lagrange3dInterpolatorVec{
            lag3_interps: lhs.iter().zip(rhs.iter()).map(|(i1,i2)| i1.clone()/i2.clone()).collect::<Vec<_>>()
        };
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T>> DivAssign<U> for Lagrange3dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    fn div_assign(&mut self, other: U) {
        for e in &mut self.lag3_interps {
            *e /= other;
        }
    }
}

// TESTS
#[cfg(test)]
pub mod lag3_tests;