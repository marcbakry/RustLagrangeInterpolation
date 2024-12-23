//! Module containing the low-level functions for the construction and the evaluation
//! of the `Lagrange1dInterpolator`.

use super::{super::utilities::{partial_sum, transpose_vec_of_vec, LagRealTrait}, LagBasicArithmetic, LagComplexTrait};

use num::one;
use num_traits::zero;

/// Evaluation of a `Lagrange1dInterpolator` using the *second form* of the barycentric formula 
/// which has a better O(k) complexity than the other approach O(k^2). This formula reads
/// 
/// (a)   `L(x) = \sum_{k=1}^n {w_k/(x - x_k) * y_k} / \sum{k=1}^n {w_k/(x - x_k)}`
/// 
/// In order to avoid a catastrophic cancellation when x == x_k, this equality is tested and
/// the corresponding y_k value is returned when true.
pub fn lag1_eval_barycentric<T,U>(xa: &Vec<T>, wa: &Vec<T>, ya: &Vec<U>, x: &T) -> U 
where 
T: LagRealTrait,
U: LagComplexTrait + LagBasicArithmetic<T> {
    let mut coef = Vec::with_capacity(xa.len());
    for i in 0..xa.len() {
        let val = *x - xa[i];
        if val.abs() > T::from(1e-12).unwrap() {
            coef.push(wa[i]/val);
        } else {
            return ya[i];
        }
    }
    return coef.iter().zip(ya.iter()).map(|(&c,&y)| y*c).sum::<U>()/coef.iter().map(|&x| x).sum::<T>();
}

/// Evaluates the Lagrange basis functions at `x` and returns the result in a vector.
pub fn lag1_eval_barycentric_basis<T,U>(wa: &Vec<T>, xa: &Vec<T>, x: &T) -> Vec<U> 
where 
T: LagRealTrait,
U: LagComplexTrait + LagBasicArithmetic<T> {
    let mut coef = Vec::with_capacity(wa.len());
    for i in 0..wa.len() {
        let val = *x -  xa[i];
        if val.abs() > T::from(1e-12).unwrap() {
            coef.push(wa[i]/val);
        } else {
            return (0..wa.len()).map(|j| if j == i {
                one::<U>()
            } else {
                zero::<U>()
            }).collect::<Vec<_>>();
        }
    }
    let den = coef.iter().map(|&x| x).sum::<T>();
    return coef.iter().map(|&cc| one::<U>()*(cc/den)).collect::<Vec<_>>();
}

/// Evaluation of the `Lagrange1dInterpolator` with the data `(xa,ya)` for a vector `x`.
pub fn lag1_eval_barycentric_vec<T,U>(xa: &Vec<T>, wa: &Vec<T>, ya: &Vec<U>, x: &Vec<T>) -> Vec<U> 
where 
T: LagRealTrait,
U: LagComplexTrait + LagBasicArithmetic<T>  {
    x.iter().map(|e| lag1_eval_barycentric(xa, wa, ya, e)).collect::<Vec<U>>()
}

/// Evaluates the Lagrange basis for a vector of values.
pub fn lag1_eval_barycentric_basis_vec<T,U>(wa: &Vec<T>, xa: &Vec<T>, x: &Vec<T>) -> Vec<Vec<U>> 
where 
T: LagRealTrait,
U: LagComplexTrait + LagBasicArithmetic<T> {
    transpose_vec_of_vec(x.iter().map(|e| lag1_eval_barycentric_basis(wa, xa, e)).collect::<Vec<_>>())
}

/// Evaluation of the first derivative of `Lagrange1dInterpolator` with the data `(xa,ya)` at `x`.
/// 
/// *Important:* This function does not perform any NaN check and may return `NaN` if `x` is
/// equal to one of the `xa`.
pub fn lag1_eval_derivative_barycentric<T,U>(xa: &Vec<T>, wa: &Vec<T>, ya: &Vec<U>, x: &T) -> U 
where 
T: LagRealTrait,
U: LagComplexTrait + LagBasicArithmetic<T>  {
    lag1_eval_barycentric(
        xa,
        wa,
        &ya.iter().enumerate().map(|(idx,&val)| val*partial_sum(xa,x,idx)).collect::<Vec<U>>(),
        x)
}

/// Compute the derivative of the Lagrange basis functions.
pub fn lag1_eval_barycentric_basis_derivative<T,U>(xa: &Vec<T>, wa: &Vec<T>, x: &T) -> Vec<U>
where 
T: LagRealTrait,
U: LagComplexTrait + LagBasicArithmetic<T>  {
    let mut output = lag1_eval_barycentric_basis(wa, xa, x); // evaluate basis
    for i in 0..wa.len() {
        output[i] *= partial_sum(xa, x, i);
    }
    return output;
}

/// Evaluation of the first derivative of `Lagrange1dInterpolator` with the data `(xa,ya)` at values in a vector `x`.
/// 
/// *Important:* This function does not perform any NaN check and may return `NaN` if `x` is
/// equal to one of the `xa`.
pub fn lag1_eval_derivative_barycentric_vec<T,U>(xa: &Vec<T>, wa: &Vec<T>, ya: &Vec<U>, x: &Vec<T>) -> Vec<U> 
where 
T: LagRealTrait,
U: LagComplexTrait + LagBasicArithmetic<T> {
    x.iter().map(|e| lag1_eval_derivative_barycentric(xa, wa, ya, e)).collect::<Vec<U>>()
}

/// Just like `lag1_eval_barycentric_basis_derivative`, but for a vector of `x`.
pub fn lag1_eval_barycentric_basis_derivative_vec<T,U>(xa: &Vec<T>, wa: &Vec<T>, x: &Vec<T>) -> Vec<Vec<U>>
where 
T: LagRealTrait,
U: LagComplexTrait + LagBasicArithmetic<T>  {
    transpose_vec_of_vec(x.iter().map(|e| lag1_eval_barycentric_basis_derivative(xa, wa, e)).collect::<Vec<_>>())
}