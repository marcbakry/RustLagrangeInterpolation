//! Module containing the low-level functions for the construction and the evaluation
//! of the `Lagrange1dInterpolator`.
use super::super::utilities::{partial_sum,LagRealTrait,LagComplexTrait};

use num_traits::zero;
use std::ops::{DivAssign, MulAssign};

/// Evaluation of the `Lagrange1dInterpolator` with the data `(xa,ya)` at `x`.
/// 
/// # Panics
/// 
/// This function will panic if two values of `xa` are identical.
pub fn lag1_eval<U,T>(xa: &Vec<T>, ya: &Vec<U>, x: &T) -> U 
where 
T: LagRealTrait,
U: LagComplexTrait + DivAssign<T> + MulAssign<T> {
    // 
    let mut ns = xa.iter().map(|e| (*e- *x).abs()).enumerate().min_by(|e1,e2| ((*e1).1).partial_cmp(&e2.1).unwrap()).map(|(idx,_)| idx).unwrap();
    let mut c = ya.clone();
    let mut d = ya.clone();
    let mut y = ya[ns];

    let n = xa.len();
    for m in 1..n {
        for i in 0..(n-m) {
            let ho = xa[i] - *x;
            let hp = xa[i+m] - *x;
            let mut w  = c[i+1] - d[i];
            let den = ho - hp;
            if den == zero::<T>() {
                panic!("Failure in lag1_eval(). Two interpolation nodes may be identical up to roundoff error");
            }
            w.div_assign(den);
            d[i] = w; d[i].mul_assign(hp);
            c[i] = w; c[i].mul_assign(ho);
        }
        if 2*ns < n-m {
            y += c[ns];
        } else {
            y += d[ns-1];
            ns -= 1;
        }
    }
    // the end
    return y;
}

/// Evaluation of the `Lagrange1dInterpolator` with the data `(xa,ya)` for a vector `x`.
/// 
/// # Panics
/// 
/// This function will panic if two values of `xa` are identical.
pub fn lag1_eval_vec<T,U>(xa: &Vec<T>, ya: &Vec<U>, x: &Vec<T>) -> Vec<U> 
where 
T: LagRealTrait,
U: LagComplexTrait + DivAssign<T> + MulAssign<T>  {
    x.iter().map(|e| lag1_eval(xa, ya, e)).collect::<Vec<U>>()
}

/// Evaluation of the first derivative of `Lagrange1dInterpolator` with the data `(xa,ya)` at `x`.
/// 
/// *Important:* This function does not perform any NaN check and may return `NaN` if `x` is
/// equal to one of the `xa`.
/// 
/// # Panics
/// 
/// This function will panic if two values of `xa` are identical.
pub fn lag1_eval_derivative<T,U>(xa: &Vec<T>, ya: &Vec<U>, x: &T) -> U 
where 
T: LagRealTrait,
U: LagComplexTrait + DivAssign<T> + MulAssign<T>  {
    lag1_eval(
        xa,
        &ya.iter().enumerate().map(|(idx,&val)| val*U::from(partial_sum(xa,x,idx)).unwrap()).collect::<Vec<U>>(),
        x)
}

/// Evaluation of the first derivative of `Lagrange1dInterpolator` with the data `(xa,ya)` at values in a vector `x`.
/// 
/// *Important:* This function does not perform any NaN check and may return `NaN` if `x` is
/// equal to one of the `xa`.
/// 
/// # Panics
/// 
/// This function will panic if two values of `xa` are identical.
pub fn lag1_eval_derivative_vec<T,U>(xa: &Vec<T>, ya: &Vec<U>, x: &Vec<T>) -> Vec<U> 
where 
T: LagRealTrait,
U: LagComplexTrait + DivAssign<T> + MulAssign<T> {
    x.iter().map(|e| lag1_eval_derivative(xa, ya, e)).collect::<Vec<U>>()
}