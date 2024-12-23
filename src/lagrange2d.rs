//! This module provide implementations for the bivariate, **gridded**, Lagrange interpolator for
//! scalar (`Lagrange2dInterpolat`or) and vector (`Lagrange2dInterpolatorVec`) real/complex
//! fields using the Rust standard library. It relies heavily on the `Vec` type.
//! 
//! By **gridded**, we mean that the interpolation nodes should be dispatched over some
//! `x1a x x2a` grid with size `n1a x n2a` using the `x2a`- major convention: let `ya` be
//! the values at the interpolation nodes, the first `n2a` values correspond to `x1a[0]`, the next `n2a` to `x1a[1]` and so on. Below, we show how we can interpolate some function `f(x1,x2)` over the unit square using a different number of nodes in each direction.
//! 
//! ```
//! use lagrange_interpolation::lagrange2d::*;
//! use lagrange_interpolation::utilities::*;
//! 
//! ...
//! 
//! let f = |x1:f64, x2: f64| f64::cos(2.0*std::f64::consts::PI*x1.powi(2))*x2.powf(1.5);
//! let (n1a,n2a) = (9,10);
//! let (a,b) = (0.0,1.0);
//! let (x1a,x2a) = (gauss_chebyshev_nodes(&n1a,&a,&b),gauss_chebyshev_nodes(&n2a,&a,&b));
//! let ya = x1a.iter().map_flat(|x1| x2a.iter().map(move |x2| f(*x1,*x2))).collect::<Vec<_>>();
//! let i2d = Lagrange2dInterpolator::new(x1a,x2a,ya);
//! let (x1,x2) = (1.0/3.0,2.0/3.0);
//! let value = i2d.eval(&x1,&x2); // interpolation at a single value (x1,x2)
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
extern crate rayon;

pub mod lag2_utilities;

use num_traits::{zero,AsPrimitive};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::fmt::{Debug,Display,Formatter,Result};
use std::ops::{Div,DivAssign,Mul,MulAssign,Add,AddAssign,Sub,SubAssign};

use super::utilities::*;
use lag2_utilities::*;

/// The `Lagrange2dInterpolator` structure holds the data for the computation of the
/// bivariate one-dimensional gridded Lagrange interpolation.
#[derive(Debug,Clone)]
pub struct Lagrange2dInterpolator<T,U> {
    x1a: Vec<T>,
    x2a: Vec<T>,
    w1a: Vec<T>,
    w2a: Vec<T>,
    ya: Vec<Vec<U>>,
    diff1_order: usize,
    diff2_order: usize
}

/// The `Lagrange2dInterpolatorVec` holds the data for the computation of the bivariate
/// multidimensional Lagrange interpolation. It contains only a `Vec<Lagrange2dInterpolator>`
/// but provides the same functionalities as `Lagrange2dInterpolator`.
#[derive(Debug,Clone)]
pub struct Lagrange2dInterpolatorVec<T,U> {
    lag2_interps: Vec<Lagrange2dInterpolator<T,U>>
}

impl<T,U> Lagrange2dInterpolatorVec<T,U> where
T: LagRealTrait, i32: AsPrimitive<T>, U: LagComplexTrait + LagBasicArithmetic<T> {
    /// Returns as `Lagrange2dInterpolatorVec` for a serie of `(x1a,x2a,ya)` interpolation data.
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
    /// let ya = vec![vec![1.0;9],vec![2.0;9]];
    /// let i2d_vec = Lagrange2dInterpolatorVec::new(x1a,x2a,ya);
    /// ```
    pub fn new(x1a: Vec<Vec<T>>, x2a: Vec<Vec<T>>, ya: Vec<Vec<U>>) -> Lagrange2dInterpolatorVec<T,U> {
        if x1a.len() != x2a.len() || x1a.len() != ya.len() {
            panic!("Error initializing the vector-field interpolator: inputs sizes do not match");
        }
        return Lagrange2dInterpolatorVec { 
            lag2_interps: ya.iter().zip(x1a.iter()).zip(x2a.iter()).map(|((y,x1),x2)| Lagrange2dInterpolator::new((*x1).clone(), (*x2).clone(), (*y).clone())).collect::<Vec<_>>()
         };
    }

    /// Evaluates a `Lagrange2dInterpolatorVec` at some `(x1,x2)`. The output is a vector
    /// containing the value returned by each inner interpolator.
    pub fn eval(&self, x1: &T, x2: &T) -> Vec<U> {
        return self.lag2_interps.iter().map(|interp| interp.eval(x1, x2)).collect::<Vec<_>>();
    }

    /// Evaluates `self` on a grid given by `x1` and `x2` following the same ordering
    /// as the interpolation grid.
    /// 
    /// # Example
    /// 
    /// ```
    /// let i2d_vec = ...;
    /// let (x1,x2) = (linrange(&10,&0.0,&1.0),gauss_chebyshev_nodes(&50,&-5.1,&-4.0));
    /// let val = i2d_vec.eval_grid(&x1,&x2);
    /// ```
    pub fn eval_grid(&self, x1: &Vec<T>, x2: &Vec<T>) -> Vec<Vec<U>>{
        return self.lag2_interps.iter().map(|interp| interp.eval_grid(x1, x2)).collect::<Vec<_>>();
    }

    /// Evaluates `self` on a set of nodes whose coordinates are given in two separate vectors.
    /// The length of `x1` and `x2` must match.
    /// 
    /// # Example
    /// 
    /// ```
    /// let i2d_vec = ...;
    /// let (x1,x2) = (linrange(&10,&0.0,&1.0),gauss_chebyshev_nodes(&10,&-5.1,&-4.0));
    /// let val = i2d_vec.eval_vec(&x1,&x2);
    /// ```
    pub fn eval_vec(&self, x1: &Vec<T>, x2: &Vec<T>) -> Vec<Vec<U>> {
        return self.lag2_interps.iter().map(|interp| interp.eval_vec(x1, x2)).collect::<Vec<_>>();
    }
    
    /// Evaluates `self` on a set of nodes whose coorinates are given in a vector 
    /// of size-two arrays.
    /// 
    /// # Example
    /// 
    /// ```
    /// let i12d_vec = ...;
    /// let (x1,x2) = (linrange(&10,&0.0,&1.0),gauss_chebyshev_nodes(&10,&-5.1,&-4.0));
    /// let val = i2d_vec.eval(
    ///         &(x1.iter().zip(x2.iter()).map(|(&x1,&x2)| [x1,x2]).collect::<Vec<_>>())
    ///     );
    /// ```
    pub fn eval_arr(&self, x: &Vec<[T;2]>) -> Vec<Vec<U>> {
        return self.lag2_interps.iter().map(|interp| interp.eval_arr(x)).collect::<Vec<_>>();
    }

    /// Evaluates `self` on a set of nodes whose coorinates are given in a vector 
    /// of size-two tuples.
    /// 
    /// # Example
    /// 
    /// ```
    /// let i1d = ...;
    /// let (x1,x2) = (linrange(&10,&0.0,&1.0),gauss_chebyshev_nodes(&10,&-5.1,&-4.0));
    /// let val = i1d.eval(
    ///         &(x1.iter().zip(x2.iter()).map(|(&x1,&x2)| (x1,x2)).collect::<Vec<_>>())
    ///     );
    /// ```
    pub fn eval_tup(&self, x: &Vec<(T,T)>) -> Vec<Vec<U>> {
        return self.lag2_interps.iter().map(|interp| interp.eval_tup(x)).collect::<Vec<_>>();
    }

    /// Parallel version of `self.eval_grid()`.
    pub fn par_eval_grid(&self, x1: &Vec<T>, x2: &Vec<T>) -> Vec<Vec<U>>{
        return self.lag2_interps.iter().map(|interp| interp.par_eval_grid(x1, x2)).collect::<Vec<_>>();
    }

    /// Parallel version of `self.eval_vec()`.
    pub fn par_eval_vec(&self, x1: &Vec<T>, x2: &Vec<T>) -> Vec<Vec<U>> {
        return self.lag2_interps.iter().map(|interp| interp.par_eval_vec(x1, x2)).collect::<Vec<_>>();
    }
    
    /// Parallel version of `self.eval_arr()`.
    pub fn par_eval_arr(&self, x: &Vec<[T;2]>) -> Vec<Vec<U>> {
        return self.lag2_interps.iter().map(|interp| interp.par_eval_arr(x)).collect::<Vec<_>>();
    }

    /// Parallel version of `self.eval_tup()`.
    pub fn par_eval_tup(&self, x: &Vec<(T,T)>) -> Vec<Vec<U>> {
        return self.lag2_interps.iter().map(|interp| interp.par_eval_tup(x)).collect::<Vec<_>>();
    }

    /// Computes the jacobian matrix of `self` as a vector of `Lagrange2dInterpolator`s. We recall that 
    /// the lines of the jacobian matrix hold the gradient of the associated component. Therefore, each
    /// entry of the output of this function holds a 2-array containing the components of the gradient.
    /// 
    /// # Example
    /// 
    /// ```
    /// let i2d_vec = ...;
    /// let jac = i2d_vec.jacobian();
    /// ```
    pub fn jacobian(&self) -> Vec<[Lagrange2dInterpolator<T,U>;2]> {
        self.lag2_interps.iter().map(|interp| [interp.differentiate_x1(),interp.differentiate_x2()]).collect::<Vec<_>>()
    }

    /// Get the inner interpolation data.
    pub fn get_inner_interpolators(&self) -> Vec<Lagrange2dInterpolator<T,U>> {
        return self.lag2_interps.clone();
    }

    /// Returns the interpolation order of each inner interpolator.
    pub fn order(&self) -> Vec<(usize,usize)> {
        self.lag2_interps.iter().map(|interp| interp.order()).collect::<Vec<_>>()
    }
    
    /// Returns the number of interpolation nodes in each direction for each inner interpolator.
    pub fn len(&self) -> Vec<(usize,usize)> {
        self.lag2_interps.iter().map(|interp| interp.len()).collect::<Vec<_>>()
    }

    /// Returns the dimension of the interpolated data.
    pub fn dim(&self) -> usize {
        return self.lag2_interps.len();
    }
}

impl<T,U> Lagrange2dInterpolator<T,U> where 
T: LagRealTrait,
i32: AsPrimitive<T>,
U: LagComplexTrait + LagBasicArithmetic<T>  {
    /// Returns a `Lagrange2dInterpolator` for some interpolation data `(x1a,x2a,ya)`.
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
    /// let ya = vec![1.0,1.1,1.2,1.3,1.4,1.5];
    /// let i2d = Lagrange2dInterpolator::new(x1a,x2,ya);
    /// ```
    pub fn new(x1a: Vec<T>, x2a: Vec<T>, ya: Vec<U>) -> Lagrange2dInterpolator<T,U> {
        // 
        if x1a.len()*x2a.len() != ya.len() {
            panic!("Invalid input: size of inputs do not match");
        }
        if x1a.len() == 0 || x2a.len()  == 0 || ya.len() == 0 {
            panic!("Invalid input: 0-sized inputs");
        }
        check_duplicate(&x1a);
        check_duplicate(&x2a);
        let idx1a = argsort(&x1a);
        let idx2a = argsort(&x2a);

        let x1a = idx1a.iter().map(|&i| x1a[i]).collect::<Vec<T>>();
        let x2a = idx2a.iter().map(|&i| x2a[i]).collect::<Vec<T>>();
        let w1a = barycentric_weights(&x1a);
        let w2a = barycentric_weights(&x2a);

        let mut ya_sorted = Vec::with_capacity(x1a.len());
        for i1a in 0..x1a.len() {
            let beg = i1a*x2a.len();
            let tmp = idx2a.iter().map(|&idx| ya[idx+beg]).collect::<Vec<U>>();
            ya_sorted.push(tmp);
        }
        let ya = idx1a.into_iter().map(|idx| ya_sorted[idx].clone()).collect::<Vec<_>>();
        // 
        return Lagrange2dInterpolator { x1a: x1a, x2a: x2a, w1a: w1a, w2a: w2a, ya: ya, diff1_order: 0, diff2_order: 0 };
    }

    /// Returns the order of the interpolating polynomial in each direction
    pub fn order(&self) -> (usize,usize) {
        (self.x1a.len()-1,self.x2a.len()-1)
    }
    
    /// Returns the number of interpolation nodes in each direction.
    pub fn len(&self) -> (usize,usize) {
        (self.x1a.len(),self.x2a.len())
    }

    /// Returns the differentiation order with respect to each of the variables.
    pub fn diff_order(&self) -> (usize,usize) {
        (self.diff1_order,self.diff2_order)
    }

    /// Get a copy of the underlying interpolation data
    pub fn get_interp_data(&self) -> (Vec<T>,Vec<T>,Vec<Vec<U>>) {
        (self.x1a.clone(),self.x2a.clone(),self.ya.clone())
    }

    /// Get a reference on the interpolation data
    pub fn get_interp_data_ref(&self) -> (&Vec<T>,&Vec<T>,&Vec<Vec<U>>) {
        (&(self.x1a),&(self.x2a),&(self.ya))
    }

    /// Evaluates the interpolator at some `(x1,x2)`.
    pub fn eval(&self,x1: &T, x2: &T) -> U {
        // lag2_eval(&self.x1a, &self.x2a, &self.ya, x1, x2)
        lag2_eval_barycentric(&self.x1a, &self.x2a, &self.w1a, &self.w2a, &self.ya, x1, x2)
    }

    /// Evaluates `self` on a grid given by `x1` and `x2` following the same ordering
    /// as the interpolation grid.
    /// 
    /// # Example
    /// 
    /// ```
    /// let i2d = ...;
    /// let (x1,x2) = (linrange(&10,&0.0,&1.0),gauss_chebyshev_nodes(&50,&-5.1,&-4.0));
    /// let val = i2d.eval_grid(&x1,&x2);
    /// ```
    pub fn eval_grid(&self, x1: &Vec<T>, x2: &Vec<T>) -> Vec<U> {
        // lag2_eval_grid(&self.x1a, &self.x2a, &self.ya, x1, x2)
        lag2_eval_grid_barycentric(&self.x1a, &self.x2a, &self.w1a, &self.w2a, &self.ya, x1, x2)
    }

    /// Evaluates `self` on a set of nodes whose coordinates are given in two separate vectors.
    /// The length of `x1` and `x2` must match.
    /// 
    /// # Example
    /// 
    /// ```
    /// let i2d = ...;
    /// let (x1,x2) = (linrange(&10,&0.0,&1.0),gauss_chebyshev_nodes(&10,&-5.1,&-4.0));
    /// let val = i2d.eval_vec(&x1,&x2);
    /// ```
    pub fn eval_vec(&self, x1: &Vec<T>, x2: &Vec<T>) -> Vec<U> {
        // lag2_eval_vec(&self.x1a, &self.x2a, &self.ya, x1, x2)
        lag2_eval_vec_barycentric(&self.x1a, &self.x2a, &self.w1a, &self.w2a, &self.ya, x1, x2)
    }
    /// Evaluates `self` on a set of nodes whose coorinates are given in a vector 
    /// of size-two arrays.
    /// 
    /// # Example
    /// 
    /// ```
    /// let i12d = ...;
    /// let (x1,x2) = (linrange(&10,&0.0,&1.0),gauss_chebyshev_nodes(&10,&-5.1,&-4.0));
    /// let val = i2d.eval(
    ///         &(x1.iter().zip(x2.iter()).map(|(&x1,&x2)| [x1,x2]).collect::<Vec<_>>())
    ///     );
    /// ```
    pub fn eval_arr(&self, x: &Vec<[T;2]>) -> Vec<U> {
        // x.iter().map(|e| lag2_eval(&self.x1a, &self.x2a, &self.ya, &e[0], &e[1])).collect::<Vec<_>>()
        x.iter().map(|e| lag2_eval_barycentric(&self.x1a, &self.x2a, &self.w1a, &self.w2a, &self.ya, &e[0], &e[1])).collect::<Vec<_>>()
    }

    /// Evaluates `self` on a set of nodes whose coorinates are given in a vector 
    /// of size-two tuples.
    /// 
    /// # Example
    /// 
    /// ```
    /// let i12d = ...;
    /// let (x1,x2) = (linrange(&10,&0.0,&1.0),gauss_chebyshev_nodes(&10,&-5.1,&-4.0));
    /// let val = i2d.eval(
    ///         &(x1.iter().zip(x2.iter()).map(|(&x1,&x2)| (x1,x2)).collect::<Vec<_>>())
    ///     );
    /// ```
    pub fn eval_tup(&self, x: &Vec<(T,T)>) -> Vec<U> {
        // x.iter().map(|e| lag2_eval(&self.x1a, &self.x2a, &self.ya, &e.0, &e.1)).collect::<Vec<_>>()
        x.iter().map(|e| lag2_eval_barycentric(&self.x1a, &self.x2a, &self.w1a, &self.w2a, &self.ya, &e.0, &e.1)).collect::<Vec<_>>()
    }

    /// Parallel version of `self.eval_grid()`.
    pub fn par_eval_grid(&self, x1: &Vec<T>, x2: &Vec<T>) -> Vec<U> {
        // (*x1).par_iter().flat_map_iter(|xx1| (*x2).iter().map(|xx2| self.eval(xx1, xx2))).collect::<Vec<_>>()
        (*x1).par_iter().flat_map_iter(|xx1| (*x2).iter().map(|xx2| self.eval(xx1, xx2))).collect::<Vec<_>>()
    }

    /// Parallel version of `self.eval_vec()`.
    pub fn par_eval_vec(&self, x1: &Vec<T>, x2: &Vec<T>) -> Vec<U> {
        (*x1).par_iter().zip_eq((*x2).par_iter()).map(|(xx1,xx2)| self.eval(xx1,xx2)).collect::<Vec<U>>()
    }

    /// Parallel version of `self.eval_arr()`.
    pub fn par_eval_arr(&self, x: &Vec<[T;2]>) -> Vec<U> {
        (*x).par_iter().map(|&xx| self.eval(&xx[0], &xx[1])).collect::<Vec<U>>()
    }

    /// Parallel version of `self.eval_tup()`.
    pub fn par_eval_tup(&self, x: &Vec<(T,T)>) -> Vec<U> {
        (*x).par_iter().map(|&(x1,x2)| self.eval(&x1,&x2)).collect::<Vec<_>>()
    }

    /// Returns the partial derivative with respect to `x1` of `self` as a new `Lagrange2dInterpolator` 
    /// on `self.len().0-1` nodes. If the length of the new interpolator falls 
    /// to 0, it returns instead a `Lagrange2dInterpolator` with a single node
    /// and the value 0.
    /// 
    /// # Example
    /// 
    /// ```
    /// let i2d = ...;
    /// let i2d_dx1 = i2d.differentiate_x1();
    /// ```
    pub fn differentiate_x1(&self) -> Lagrange2dInterpolator<T, U> {
        // 
        let (x1a,x2a, mut ya) = self.get_interp_data();
        let new_diff1_order = self.diff1_order+1;
        let (w1a,w2a) = (barycentric_weights(&x1a),barycentric_weights(&x2a));
        // 
        if self.x1a.len()-1 == 0 {
            ya.iter_mut().for_each(|y| y.iter_mut().for_each(|e| *e = zero::<U>()));

            return Lagrange2dInterpolator {
                x1a: x1a,
                x2a: x2a,
                w1a: w1a,
                w2a: w2a,
                ya: ya,
                diff1_order: new_diff1_order,
                diff2_order: self.diff2_order
            };
        } else {
            let x1a_new = midpoints(&x1a);
            let ya_new = lag2_diff1_grid_barycentric(&x1a, &x2a, &w1a, &w2a, &ya, &x1a_new, &x2a);
            let mut output = Lagrange2dInterpolator::new(x1a_new,x2a,ya_new);
            output.diff1_order = new_diff1_order;
            output.diff2_order = self.diff2_order;
            return output;
        }
    }

    /// Returns the partial derivative with respect to `x2` of `self` as a new `Lagrange2dInterpolator` 
    /// on `self.len().1-1` nodes. If the length of the new interpolator falls 
    /// to 0, it returns instead a `Lagrange2dInterpolator` with a single node
    /// and the value 0.
    /// 
    /// # Example
    /// 
    /// ```
    /// let i2d = ...;
    /// let i2d_dx2 = i2d.differentiate_x2();
    /// ```
    pub fn differentiate_x2(&self) -> Lagrange2dInterpolator<T, U> {
        // 
        let (x1a,x2a, mut ya) = self.get_interp_data();
        let (w1a,w2a) = (barycentric_weights(&x1a),barycentric_weights(&x2a));
        let new_diff2_order = self.diff2_order+1;
        // 
        if self.x2a.len()-1 == 0 {
            ya.iter_mut().for_each(|y| y.iter_mut().for_each(|e| *e = zero::<U>()));


            return Lagrange2dInterpolator {
                x1a: x1a,
                x2a: x2a,
                w1a: w1a,
                w2a: w2a,
                ya: ya,
                diff1_order: self.diff1_order,
                diff2_order: new_diff2_order
            };
        } else {
            let x2a_new = midpoints(&x2a);
            let ya_new = lag2_diff2_grid_barycentric(&x1a, &x2a, &w1a, &w2a, &ya, &x1a, &x2a_new);
            let mut output = Lagrange2dInterpolator::new(x1a,x2a_new,ya_new);
            output.diff1_order = self.diff1_order;
            output.diff2_order = new_diff2_order;
            return output;
        }
    }

    /// Computes the gradient of the interpolator and returns it as a
    /// size-2 array.
    pub fn gradient(&self) -> [Lagrange2dInterpolator<T,U>;2] {
        return [self.differentiate_x1(),self.differentiate_x2()];
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T> > Display for Lagrange2dInterpolator<T,U> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f,"Lagrange 2d gridded interpolator:\n- length = {} x {}\n- differentiation order = ({},{})",self.x1a.len(),self.x2a.len(),self.diff1_order,self.diff2_order)
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T> > Add<U> for Lagrange2dInterpolator<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange2dInterpolator<T,U>;
    fn add(self, rhs: U) -> Self::Output {
        let (x1a,x2a,ya) = self.get_interp_data();
        let new_ya = ya.iter().flat_map(|y| y.iter()).map(|&e| e+rhs).collect::<Vec<_>>();
        return Lagrange2dInterpolator::new(x1a, x2a, new_ya);
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T> > Add<Lagrange2dInterpolator<T,U>> for Lagrange2dInterpolator<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange2dInterpolator<T,U>;
    fn add(self, rhs: Lagrange2dInterpolator<T,U>) -> Self::Output {
        let (x1a_lhs,x2a_lhs,ya_lhs) = self.get_interp_data();
        let (x1a_rhs,x2a_rhs,ya_rhs) = rhs.get_interp_data();
        
        let is_same_x1a = x1a_lhs.iter().zip(x1a_rhs.iter()).any(|(&a,&b)| (a-b).abs() > T::from(1e-12).unwrap()) && (x1a_lhs.len() == x1a_rhs.len());
        let is_same_x2a = x2a_lhs.iter().zip(x2a_rhs.iter()).any(|(&a,&b)| (a-b).abs() > T::from(1e-12).unwrap()) && (x2a_lhs.len() == x1a_rhs.len());

        // flatten y
        let ya_lhs = ya_lhs.iter().flat_map(|e| e.iter()).map(|&e| e).collect::<Vec<_>>();
        let ya_rhs = ya_rhs.iter().flat_map(|e| e.iter()).map(|&e| e).collect::<Vec<_>>();

        let (x1a_new,x2a_new,ya_new) = if is_same_x1a && is_same_x2a {
            (x1a_lhs,x2a_lhs,ya_lhs.iter().zip(ya_rhs.iter()).map(|(&a,&b)| a+b).collect::<Vec<_>>())
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
            let ya_lhs = self.eval_grid(&x1a_new, &x2a_new);
            let ya_rhs = rhs.eval_grid(&x1a_new, &x2a_new);
            let ya_new = ya_lhs.iter().zip(ya_rhs.iter()).map(|(&a,&b)| a+b).collect::<Vec<_>>();
            (x1a_new,x2a_new,ya_new)
        };
        return Lagrange2dInterpolator::new(x1a_new,x2a_new,ya_new);
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T> > AddAssign<U> for Lagrange2dInterpolator<T,U> where i32: AsPrimitive<T> {
    fn add_assign(&mut self, rhs: U) {
        for y in &mut self.ya {
            for yy in y {
                *yy = *yy + rhs;
            }
        }
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T> > Sub<U> for Lagrange2dInterpolator<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange2dInterpolator<T,U>;
    fn sub(self, rhs: U) -> Self::Output {
        let (x1a,x2a,ya) = self.get_interp_data();
        let new_ya = ya.iter().flat_map(|y| y.iter()).map(|&e| e - rhs).collect::<Vec<_>>();
        return Lagrange2dInterpolator::new(x1a, x2a, new_ya);
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T> > Sub<Lagrange2dInterpolator<T,U>> for Lagrange2dInterpolator<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange2dInterpolator<T,U>;
    fn sub(self, rhs: Lagrange2dInterpolator<T,U>) -> Self::Output {
        let (x1a_lhs,x2a_lhs,ya_lhs) = self.get_interp_data();
        let (x1a_rhs,x2a_rhs,ya_rhs) = rhs.get_interp_data();
        
        let is_same_x1a = x1a_lhs.iter().zip(x1a_rhs.iter()).any(|(&a,&b)| (a-b).abs() > T::from(1e-12).unwrap()) && (x1a_lhs.len() == x1a_rhs.len());
        let is_same_x2a = x2a_lhs.iter().zip(x2a_rhs.iter()).any(|(&a,&b)| (a-b).abs() > T::from(1e-12).unwrap()) && (x2a_lhs.len() == x1a_rhs.len());

        // flatten y
        let ya_lhs = ya_lhs.iter().flat_map(|e| e.iter()).map(|&e| e).collect::<Vec<_>>();
        let ya_rhs = ya_rhs.iter().flat_map(|e| e.iter()).map(|&e| e).collect::<Vec<_>>();

        let (x1a_new,x2a_new,ya_new) = if is_same_x1a && is_same_x2a {
            (x1a_lhs,x2a_lhs,ya_lhs.iter().zip(ya_rhs.iter()).map(|(&a,&b)| a-b).collect::<Vec<_>>())
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
            let ya_lhs = self.eval_grid(&x1a_new, &x2a_new);
            let ya_rhs = rhs.eval_grid(&x1a_new, &x2a_new);
            let ya_new = ya_lhs.iter().zip(ya_rhs.iter()).map(|(&a,&b)| a-b).collect::<Vec<_>>();
            (x1a_new,x2a_new,ya_new)
        };
        return Lagrange2dInterpolator::new(x1a_new,x2a_new,ya_new);
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T> > SubAssign<U> for Lagrange2dInterpolator<T,U> where i32: AsPrimitive<T> {
    fn sub_assign(&mut self, rhs: U) {
        for y in &mut self.ya {
            for yy in y {
                *yy = *yy - rhs;
            }
        }
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T> > Mul<U> for Lagrange2dInterpolator<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange2dInterpolator<T,U>;
    fn mul(self, rhs: U) -> Self::Output {
        let (x1a,x2a,ya) = self.get_interp_data();
        let new_ya = ya.iter().flat_map(|y| y.iter()).map(|&e| e*rhs).collect::<Vec<_>>();
        return Lagrange2dInterpolator::new(x1a, x2a, new_ya);
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T> > Mul<Lagrange2dInterpolator<T,U>> for Lagrange2dInterpolator<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange2dInterpolator<T,U>;
    fn mul(self, rhs: Lagrange2dInterpolator<T,U>) -> Self::Output {
        let (x1a_lhs,x2a_lhs,ya_lhs) = self.get_interp_data();
        let (x1a_rhs,x2a_rhs,ya_rhs) = rhs.get_interp_data();
        
        let is_same_x1a = x1a_lhs.iter().zip(x1a_rhs.iter()).any(|(&a,&b)| (a-b).abs() > T::from(1e-12).unwrap()) && (x1a_lhs.len() == x1a_rhs.len());
        let is_same_x2a = x2a_lhs.iter().zip(x2a_rhs.iter()).any(|(&a,&b)| (a-b).abs() > T::from(1e-12).unwrap()) && (x2a_lhs.len() == x1a_rhs.len());

        // flatten y
        let ya_lhs = ya_lhs.iter().flat_map(|e| e.iter()).map(|&e| e).collect::<Vec<_>>();
        let ya_rhs = ya_rhs.iter().flat_map(|e| e.iter()).map(|&e| e).collect::<Vec<_>>();

        let (x1a_new,x2a_new,ya_new) = if is_same_x1a && is_same_x2a {
            (x1a_lhs,x2a_lhs,ya_lhs.iter().zip(ya_rhs.iter()).map(|(&a,&b)| a*b).collect::<Vec<_>>())
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
            let ya_lhs = self.eval_grid(&x1a_new, &x2a_new);
            let ya_rhs = rhs.eval_grid(&x1a_new, &x2a_new);
            let ya_new = ya_lhs.iter().zip(ya_rhs.iter()).map(|(&a,&b)| a*b).collect::<Vec<_>>();
            (x1a_new,x2a_new,ya_new)
        };
        return Lagrange2dInterpolator::new(x1a_new,x2a_new,ya_new);
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T> > MulAssign<U> for Lagrange2dInterpolator<T,U> where i32: AsPrimitive<T> {
    fn mul_assign(&mut self, rhs: U) {
        for y in &mut self.ya {
            for yy in y {
                *yy = *yy * rhs;
            }
        }
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T> > Div<U> for Lagrange2dInterpolator<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange2dInterpolator<T,U>;
    fn div(self, rhs: U) -> Self::Output {
        let (x1a,x2a,ya) = self.get_interp_data();
        let new_ya = ya.iter().flat_map(|y| y.iter()).map(|&e| e/rhs).collect::<Vec<_>>();
        return Lagrange2dInterpolator::new(x1a, x2a, new_ya);
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T> > Div<Lagrange2dInterpolator<T,U>> for Lagrange2dInterpolator<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange2dInterpolator<T,U>;
    fn div(self, rhs: Lagrange2dInterpolator<T,U>) -> Self::Output {
        let (x1a_lhs,x2a_lhs,ya_lhs) = self.get_interp_data();
        let (x1a_rhs,x2a_rhs,ya_rhs) = rhs.get_interp_data();
        
        let is_same_x1a = x1a_lhs.iter().zip(x1a_rhs.iter()).any(|(&a,&b)| (a-b).abs() > T::from(1e-12).unwrap()) && (x1a_lhs.len() == x1a_rhs.len());
        let is_same_x2a = x2a_lhs.iter().zip(x2a_rhs.iter()).any(|(&a,&b)| (a-b).abs() > T::from(1e-12).unwrap()) && (x2a_lhs.len() == x1a_rhs.len());

        // flatten y
        let ya_lhs = ya_lhs.iter().flat_map(|e| e.iter()).map(|&e| e).collect::<Vec<_>>();
        let ya_rhs = ya_rhs.iter().flat_map(|e| e.iter()).map(|&e| e).collect::<Vec<_>>();

        let (x1a_new,x2a_new,ya_new) = if is_same_x1a && is_same_x2a {
            (x1a_lhs,x2a_lhs,ya_lhs.iter().zip(ya_rhs.iter()).map(|(&a,&b)| a/b).collect::<Vec<_>>())
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
            let ya_lhs = self.eval_grid(&x1a_new, &x2a_new);
            let ya_rhs = rhs.eval_grid(&x1a_new, &x2a_new);
            let ya_new = ya_lhs.iter().zip(ya_rhs.iter()).map(|(&a,&b)| a/b).collect::<Vec<_>>();
            (x1a_new,x2a_new,ya_new)
        };
        return Lagrange2dInterpolator::new(x1a_new,x2a_new,ya_new);
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T> > DivAssign<U> for Lagrange2dInterpolator<T,U> where i32: AsPrimitive<T> {
    fn div_assign(&mut self, rhs: U) {
        for y in &mut self.ya {
            for yy in y {
                *yy = *yy / rhs;
            }
        }
    }
}

// implementation of the basic operators for Lagrange2dInterpolatorVec
impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T> > Add<U> for Lagrange2dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange2dInterpolatorVec<T,U>;

    fn add(self, rhs: U) -> Self::Output {
        return Lagrange2dInterpolatorVec{
            lag2_interps: self.get_inner_interpolators().iter().map(|interp| interp.clone() + rhs).collect::<Vec<_>>()
        };
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T> > Add<Lagrange2dInterpolatorVec<T,U>> for Lagrange2dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange2dInterpolatorVec<T,U>;

    fn add(self, rhs: Lagrange2dInterpolatorVec<T,U>) -> Self::Output {
        let lhs = self.get_inner_interpolators();
        let rhs = rhs.get_inner_interpolators();
        return Lagrange2dInterpolatorVec{
            lag2_interps: lhs.iter().zip(rhs.iter()).map(|(i1,i2)| i1.clone()+i2.clone()).collect::<Vec<_>>()
        };
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T> > AddAssign<U> for Lagrange2dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    fn add_assign(&mut self, other: U) {
        for e in &mut self.lag2_interps {
            *e += other;
        }
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T> > Sub<U> for Lagrange2dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange2dInterpolatorVec<T,U>;

    fn sub(self, rhs: U) -> Self::Output {
        return Lagrange2dInterpolatorVec{
            lag2_interps: self.get_inner_interpolators().iter().map(|interp| interp.clone() - rhs).collect::<Vec<_>>()
        };
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T> > Sub<Lagrange2dInterpolatorVec<T,U>> for Lagrange2dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange2dInterpolatorVec<T,U>;

    fn sub(self, rhs: Lagrange2dInterpolatorVec<T,U>) -> Self::Output {
        let lhs = self.get_inner_interpolators();
        let rhs = rhs.get_inner_interpolators();
        return Lagrange2dInterpolatorVec{
            lag2_interps: lhs.iter().zip(rhs.iter()).map(|(i1,i2)| i1.clone()-i2.clone()).collect::<Vec<_>>()
        };
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T> > SubAssign<U> for Lagrange2dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    fn sub_assign(&mut self, other: U) {
        for e in &mut self.lag2_interps {
            *e -= other;
        }
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T> > Mul<U> for Lagrange2dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange2dInterpolatorVec<T,U>;

    fn mul(self, rhs: U) -> Self::Output {
        return Lagrange2dInterpolatorVec{
            lag2_interps: self.get_inner_interpolators().iter().map(|interp| interp.clone()*rhs).collect::<Vec<_>>()
        };
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T> > Mul<Lagrange2dInterpolatorVec<T,U>> for Lagrange2dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange2dInterpolatorVec<T,U>;

    fn mul(self, rhs: Lagrange2dInterpolatorVec<T,U>) -> Self::Output {
        let lhs = self.get_inner_interpolators();
        let rhs = rhs.get_inner_interpolators();
        return Lagrange2dInterpolatorVec{
            lag2_interps: lhs.iter().zip(rhs.iter()).map(|(i1,i2)| i1.clone()*i2.clone()).collect::<Vec<_>>()
        };
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T> > MulAssign<U> for Lagrange2dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    fn mul_assign(&mut self, other: U) {
        for e in &mut self.lag2_interps {
            *e *= other;
        }
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T> > Div<U> for Lagrange2dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange2dInterpolatorVec<T,U>;

    fn div(self, rhs: U) -> Self::Output {
        return Lagrange2dInterpolatorVec{
            lag2_interps: self.get_inner_interpolators().iter().map(|interp| interp.clone()/rhs).collect::<Vec<_>>()
        };
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T> > Div<Lagrange2dInterpolatorVec<T,U>> for Lagrange2dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange2dInterpolatorVec<T,U>;

    fn div(self, rhs: Lagrange2dInterpolatorVec<T,U>) -> Self::Output {
        let lhs = self.get_inner_interpolators();
        let rhs = rhs.get_inner_interpolators();
        return Lagrange2dInterpolatorVec{
            lag2_interps: lhs.iter().zip(rhs.iter()).map(|(i1,i2)| i1.clone()/i2.clone()).collect::<Vec<_>>()
        };
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + LagBasicArithmetic<T> > DivAssign<U> for Lagrange2dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    fn div_assign(&mut self, other: U) {
        for e in &mut self.lag2_interps {
            *e /= other;
        }
    }
}

// TESTS
#[cfg(test)]
pub mod lag2_tests;