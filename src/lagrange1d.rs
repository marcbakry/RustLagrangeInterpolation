extern crate num_traits;
extern crate rayon;

use num_traits::zero;
use num_traits::AsPrimitive;
use std::fmt::{Debug,Display,Formatter,Result};
use std::ops::{Div,DivAssign,Mul,MulAssign,Add,AddAssign,Sub,SubAssign};
use rayon::prelude::*;

use super::utilities::*;
use super::lag1_utilities::*;

#[derive(Debug,Clone)]
pub struct Lagrange1dInterpolator<T,U> {
    xa: Vec<T>,
    ya: Vec<U>,
    diff_order: usize
}

#[derive(Debug,Clone)]
pub struct Lagrange1dInterpolatorVec<T,U> {
    lag1_interps: Vec<Lagrange1dInterpolator<T,U>>
}

impl<T,U> Lagrange1dInterpolatorVec<T,U> where 
T: LagRealTrait, i32: AsPrimitive<T>, U: LagComplexTrait + DivAssign<T> + MulAssign<T> {
    pub fn new(xa: Vec<Vec<T>>, ya: Vec<Vec<U>>) -> Lagrange1dInterpolatorVec<T,U> {
        if xa.len() != ya.len() {
            panic!("Error initializing the vector-field interpolator: inputs sizes do not match");
        }
        return Lagrange1dInterpolatorVec { 
            lag1_interps: ya.iter().zip(xa.iter()).map(|(y,x)| Lagrange1dInterpolator::new((*x).clone(), (*y).clone())).collect::<Vec<_>>() 
        };
    }

    pub fn eval(&self, x: &T) -> Vec<U> {
        return self.lag1_interps.iter().map(|interp| interp.eval(x)).collect::<Vec<U>>();
    }

    pub fn eval_vec(&self, x: &Vec<T>) -> Vec<Vec<U>> {
        // For each x-value, returns the value of all inner interpolators
        return x.iter().map(|x| self.eval(x)).collect::<Vec<_>>();
    }
    
    pub fn par_eval_vec(&self, x: &Vec<T>) -> Vec<Vec<U>>{
        return (*x).par_iter().map(|xx| self.eval(xx)).collect::<Vec<_>>();
    }

    pub fn differentiate(&self) -> Lagrange1dInterpolatorVec<T,U> {
        return Lagrange1dInterpolatorVec {
            lag1_interps: self.lag1_interps.iter().map(|interp| interp.differentiate()).collect::<Vec<_>>()
        };
    }

    pub fn get_inner_interpolators(&self) -> Vec<Lagrange1dInterpolator<T, U>> {
        return self.lag1_interps.clone();
    }

    pub fn order(&self) -> Vec<usize> {
        return self.lag1_interps.iter().map(|interp| interp.order()).collect::<Vec<_>>();
    }

    pub fn len(&self) -> Vec<usize> {
        return self.lag1_interps.iter().map(|interp| interp.len()).collect::<Vec<_>>();
    }

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
        let mut new_self = self.clone();
        for e in &mut new_self.ya {
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
        let mut new_self = self.clone();
        for e in &mut new_self.ya {
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
        let mut new_self = self.clone();
        for e in &mut new_self.ya {
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
        let mut new_self = self.clone();
        for e in &mut new_self.ya {
            *e = (*e)/other;
        }
    }
}

// TESTS
#[cfg(test)]
mod lagrange1d_tests {
    use super::*;
    use std::f64::consts::PI;
    use num::complex::*;

    #[test]
    fn lag1_real_interpolation() {
        // function and first derivative
        let f = |x: f64| f64::cos(2.0*PI*x.powi(2));
        let df = |x: f64| -4.0*PI*x*f64::sin(2.0*PI*x.powi(2));
        // interpolation data
        let (a,b) = (0.0,1.0);
        let na = 20;
        let xa = gauss_chebyshev_nodes(&na, &a, &b);
        let ya = xa.iter().map(|&x| f(x)).collect::<Vec<f64>>();
        let lag1_f = Lagrange1dInterpolator::new(xa,ya);
        let lag1_df = lag1_f.differentiate();

        // interpolated data
        let ni = 100;
        let stpi = (b-a)/(ni-1) as f64;
        let xi = (0..ni).map(|i| i as f64*stpi).collect::<Vec<f64>>();
        let yi_f = lag1_f.eval_vec(&xi);
        let yi_df = lag1_df.eval_vec(&xi);

        // reference data
        let yref_f = xi.iter().map(|&e| f(e)).collect::<Vec<f64>>();
        let yref_df = xi.iter().map(|&e| df(e)).collect::<Vec<f64>>();

        // check accuracy with the maximum of the absolute error
        let err_f = yi_f.iter().zip(yref_f.iter()).map(|(ei,ef)| (ei-ef).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap();
        let err_df = yi_df.iter().zip(yref_df.iter()).map(|(ei,ef)| (ei-ef).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap();

        println!("Error 0-th order derivative = {}",err_f);
        println!("Error 1-st order derivative = {}",err_df);

        assert_eq!(lag1_f.order(),na-1);
        assert_eq!(lag1_df.order(),na-2);
        assert_eq!(lag1_f.diff_order(),0);
        assert_eq!(lag1_df.diff_order(),1);
        assert!(err_f < 1e-6);
        assert!(err_df < 1e-3);
    }

    #[test]
    fn lag1_complex_interpolation() {
        // function and first derivative
        let f = |x: f64| Complex::new(f64::cos(2.0*PI*x.powi(2)),f64::sin(2.0*PI*x.powi(2)));
        let df = |x: f64| Complex::new(-4.0*PI*x*f64::sin(2.0*PI*x.powi(2)), 4.0*PI*x*f64::cos(2.0*PI*x.powi(2)));
        // interpolation data
        let (a,b) = (0.0,1.0);
        let na = 20;
        let xa = gauss_chebyshev_nodes(&na, &a, &b);
        let ya = xa.iter().map(|&x| f(x)).collect::<Vec<_>>();
        let lag1_f = Lagrange1dInterpolator::new(xa,ya);
        let lag1_df = lag1_f.differentiate();

        // interpolated data
        let ni = 100;
        let stpi = (b-a)/(ni-1) as f64;
        let xi = (0..ni).map(|i| i as f64*stpi).collect::<Vec<f64>>();
        let yi_f = lag1_f.eval_vec(&xi);
        let yi_df = lag1_df.eval_vec(&xi);

        // reference data
        let yref_f = xi.iter().map(|&e| f(e)).collect::<Vec<_>>();
        let yref_df = xi.iter().map(|&e| df(e)).collect::<Vec<_>>();

        // check accuracy with the maximum of the absolute error
        let err_f = yi_f.iter().zip(yref_f.iter()).map(|(ei,ef)| (ei-ef).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap();
        let err_df = yi_df.iter().zip(yref_df.iter()).map(|(ei,ef)| (ei-ef).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap();

        println!("Error 0-th order derivative = {}",err_f);
        println!("Error 1-st order derivative = {}",err_df);

        assert_eq!(lag1_f.order(),na-1);
        assert_eq!(lag1_df.order(),na-2);
        assert_eq!(lag1_f.diff_order(),0);
        assert_eq!(lag1_df.diff_order(),1);
        assert!(err_f < 2.0*1e-6);
        assert!(err_df < 2.0*1e-3);
    }

    #[test]
    #[should_panic]
    pub fn lag1_input_size_mismatch() {
        let xa: Vec<f64> = vec![1.0,2.0,3.0];
        let mut ya = vec![1.0; 3];
        ya.pop();

        let _lag1_f = Lagrange1dInterpolator::new(xa,ya);
    }

    #[test]
    #[should_panic]
    pub fn lag1_null_size_xa_input() {
        let xa: Vec<f64> = Vec::new();
        let ya = vec![0.0; 3];
        let _lag1_f = Lagrange1dInterpolator::new(xa,ya);
    }
    
    #[test]
    #[should_panic]
    pub fn lag1_null_size_ya_input() {
        let xa: Vec<f64> = vec![1.0,2.0,3.0];
        let ya: Vec<f64> = Vec::new();
        let _lag1_f = Lagrange1dInterpolator::new(xa,ya);
    }

    #[test]
    #[should_panic]
    pub fn lag1_duplicate_entries() {
        let xa: Vec<f64> = vec![0.0; 3];
        let ya: Vec<f64> = xa.clone();
        let _lag1_f = Lagrange1dInterpolator::new(xa,ya);
    }

    #[test]
    pub fn lag1_add_operators() {
        // 
        let f1 = |x: f64| f64::cos(2.0*PI*x);
        let f2 = |x: f64| x.powf(1.5);
        let f1plusf2 = |x: f64| f1(x) + f2(x);
        let (a,b) = (0.0,1.0);
        let (nm,np) = (9,10);
        let (xm,xp) = (gauss_chebyshev_nodes(&nm, &a, &b),gauss_chebyshev_nodes(&np, &a, &b));
        // 
        let ni = 100;
        let xi = linspace(&ni, &a, &b);

        // test add with same xa == xp
        let (y1,y2): (Vec<f64>,Vec<f64>) = xp.iter().map(|&x| (f1(x),f2(x))).unzip();
        let y12 = xp.iter().map(|&x| f1plusf2(x)).collect::<Vec<_>>();
        let interp1 = Lagrange1dInterpolator::new(xp.clone(), y1);
        let interp2 = Lagrange1dInterpolator::new(xp.clone(), y2);
        let interp12 = interp1+interp2;
        let interp12_ref = Lagrange1dInterpolator::new(xp.clone(),y12);
        let (y_cal,y_ref) = (interp12.eval_vec(&xi),interp12_ref.eval_vec(&xi));
        assert!(y_cal.iter().zip(y_ref.iter()).all(|(&a,&b)| (a-b).abs() < 1e-12));

        // test add with different xa
        let y1 = xm.iter().map(|&x| f1(x)).collect::<Vec<_>>();
        let y2 = xp.iter().map(|&x| f2(x)).collect::<Vec<_>>();
        let y12 = xp.iter().map(|&x| f1plusf2(x)).collect::<Vec<_>>();
        let interp1 = Lagrange1dInterpolator::new(xm.clone(), y1);
        let interp2 = Lagrange1dInterpolator::new(xp.clone(), y2);
        let interp12 = interp2+interp1;
        let interp12_ref = Lagrange1dInterpolator::new(xp.clone(),y12);
        let (y_cal,y_ref) = (interp12.eval_vec(&xi),interp12_ref.eval_vec(&xi));
        assert!(y_cal.iter().zip(y_ref.iter()).all(|(&a,&b)| (a-b).abs() < 1e-4));
    }
    
    #[test]
    pub fn lag1_sub_operators() {
        // 
        let f1 = |x: f64| f64::cos(2.0*PI*x);
        let f2 = |x: f64| x.powf(1.5);
        let f1plusf2 = |x: f64| f1(x) - f2(x);
        let (a,b) = (0.0,1.0);
        let (nm,np) = (9,10);
        let (xm,xp) = (gauss_chebyshev_nodes(&nm, &a, &b),gauss_chebyshev_nodes(&np, &a, &b));
        // 
        let ni = 100;
        let xi = linspace(&ni, &a, &b);

        // test add with same xa == xp
        let (y1,y2): (Vec<f64>,Vec<f64>) = xp.iter().map(|&x| (f1(x),f2(x))).unzip();
        let y12 = xp.iter().map(|&x| f1plusf2(x)).collect::<Vec<_>>();
        let interp1 = Lagrange1dInterpolator::new(xp.clone(), y1);
        let interp2 = Lagrange1dInterpolator::new(xp.clone(), y2);
        let interp12 = interp1-interp2;
        let interp12_ref = Lagrange1dInterpolator::new(xp.clone(),y12);
        let (y_cal,y_ref) = (interp12.eval_vec(&xi),interp12_ref.eval_vec(&xi));
        assert!(y_cal.iter().zip(y_ref.iter()).all(|(&a,&b)| (a-b).abs() < 1e-12));

        // test add with different xa
        let y1 = xm.iter().map(|&x| f1(x)).collect::<Vec<_>>();
        let y2 = xp.iter().map(|&x| f2(x)).collect::<Vec<_>>();
        let y12 = xp.iter().map(|&x| f1plusf2(x)).collect::<Vec<_>>();
        let interp1 = Lagrange1dInterpolator::new(xm.clone(), y1);
        let interp2 = Lagrange1dInterpolator::new(xp.clone(), y2);
        let interp12 = interp1-interp2;
        let interp12_ref = Lagrange1dInterpolator::new(xp.clone(),y12);
        let (y_cal,y_ref) = (interp12.eval_vec(&xi),interp12_ref.eval_vec(&xi));
        assert!(y_cal.iter().zip(y_ref.iter()).all(|(&a,&b)| (a-b).abs() < 1e-4));
    }
    
    #[test]
    pub fn lag1_mul_operators() {
        // 
        let f1 = |x: f64| f64::cos(2.0*PI*x);
        let f2 = |x: f64| x.powf(1.5);
        let f1plusf2 = |x: f64| f1(x)*f2(x);
        let (a,b) = (0.0,1.0);
        let (nm,np) = (9,10);
        let (xm,xp) = (gauss_chebyshev_nodes(&nm, &a, &b),gauss_chebyshev_nodes(&np, &a, &b));
        // 
        let ni = 100;
        let xi = linspace(&ni, &a, &b);

        // test add with same xa == xp
        let (y1,y2): (Vec<f64>,Vec<f64>) = xp.iter().map(|&x| (f1(x),f2(x))).unzip();
        let y12 = xp.iter().map(|&x| f1plusf2(x)).collect::<Vec<_>>();
        let interp1 = Lagrange1dInterpolator::new(xp.clone(), y1);
        let interp2 = Lagrange1dInterpolator::new(xp.clone(), y2);
        let interp12 = interp1*interp2;
        let interp12_ref = Lagrange1dInterpolator::new(xp.clone(),y12);
        let (y_cal,y_ref) = (interp12.eval_vec(&xi),interp12_ref.eval_vec(&xi));
        assert!(y_cal.iter().zip(y_ref.iter()).all(|(&a,&b)| (a-b).abs() < 1e-12));

        // test add with different xa
        let y1 = xm.iter().map(|&x| f1(x)).collect::<Vec<_>>();
        let y2 = xp.iter().map(|&x| f2(x)).collect::<Vec<_>>();
        let y12 = xp.iter().map(|&x| f1plusf2(x)).collect::<Vec<_>>();
        let interp1 = Lagrange1dInterpolator::new(xm.clone(), y1);
        let interp2 = Lagrange1dInterpolator::new(xp.clone(), y2);
        let interp12 = interp2*interp1;
        let interp12_ref = Lagrange1dInterpolator::new(xp.clone(),y12);
        let (y_cal,y_ref) = (interp12.eval_vec(&xi),interp12_ref.eval_vec(&xi));
        assert!(y_cal.iter().zip(y_ref.iter()).all(|(&a,&b)| (a-b).abs() < 1e-4));
    }

    #[test]
    pub fn lag1_div_operators() {
        // 
        let f1 = |x: f64| f64::cos(2.0*PI*x);
        let f2 = |x: f64| 1.0+x.powf(1.5);
        let f1plusf2 = |x: f64| f1(x)/f2(x);
        let (a,b) = (0.0,1.0);
        let (nm,np) = (9,10);
        let (xm,xp) = (gauss_chebyshev_nodes(&nm, &a, &b),gauss_chebyshev_nodes(&np, &a, &b));
        // 
        let ni = 100;
        let xi = linspace(&ni, &a, &b);

        // test add with same xa == xp
        let (y1,y2): (Vec<f64>,Vec<f64>) = xp.iter().map(|&x| (f1(x),f2(x))).unzip();
        let y12 = xp.iter().map(|&x| f1plusf2(x)).collect::<Vec<_>>();
        let interp1 = Lagrange1dInterpolator::new(xp.clone(), y1);
        let interp2 = Lagrange1dInterpolator::new(xp.clone(), y2);
        let interp12 = interp1/interp2;
        let interp12_ref = Lagrange1dInterpolator::new(xp.clone(),y12);
        let (y_cal,y_ref) = (interp12.eval_vec(&xi),interp12_ref.eval_vec(&xi));
        assert!(y_cal.iter().zip(y_ref.iter()).all(|(&a,&b)| (a-b).abs() < 1e-12));

        // test add with different xa
        let y1 = xm.iter().map(|&x| f1(x)).collect::<Vec<_>>();
        let y2 = xp.iter().map(|&x| f2(x)).collect::<Vec<_>>();
        let y12 = xp.iter().map(|&x| f1plusf2(x)).collect::<Vec<_>>();
        let interp1 = Lagrange1dInterpolator::new(xm.clone(), y1);
        let interp2 = Lagrange1dInterpolator::new(xp.clone(), y2);
        let interp12 = interp1/interp2;
        let interp12_ref = Lagrange1dInterpolator::new(xp.clone(),y12);
        let (y_cal,y_ref) = (interp12.eval_vec(&xi),interp12_ref.eval_vec(&xi));
        assert!(y_cal.iter().zip(y_ref.iter()).all(|(&a,&b)| (a-b).abs() < 1e-4));
    }
}