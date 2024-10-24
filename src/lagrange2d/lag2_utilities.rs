//! Module containing the low-level functions for the construction and the evaluation
//! of the `Lagrange2dInterpolator`.
use super::super::utilities::{partial_sum,LagRealTrait,LagComplexTrait}; 
use crate::lagrange1d::lag1_utilities::*;

use num_traits::zero;

/// Evaluation of the `Lagrange2dInterpolator` with the data `(x1a,x2a,ya)` at `(x1,x2)`.
/// The data in `ya` should be sorted such that `ya[0]` contains the values 
/// along `x2a` for the first value in `x1a`, and so on.
/// 
/// # Panics
/// 
/// This function will panic if two values in `x1a` or `x2a` are identical.
pub fn lag2_eval<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, ya: &Vec<Vec<U>>, x1: &T, x2: &T) -> U 
where 
T: LagRealTrait,
U: LagComplexTrait<T> {
    lag1_eval(x1a,&ya.iter().map(|e| lag1_eval(x2a,e,x2)).collect::<Vec<U>>(),x1)
}

/// Same as `lag2_eval` but based on the barycentric formula. See also `lag1_eval_barycentric`.
pub fn lag2_eval_barycentric<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, w1a: &Vec<T>, w2a: &Vec<T>, ya: &Vec<Vec<U>>, x1: &T, x2: &T) -> U 
where 
T: LagRealTrait,
U: LagComplexTrait<T> {
    lag1_eval_barycentric(x1a, w1a, &ya.iter().map(|e| lag1_eval_barycentric(x2a, w2a,e,x2)).collect::<Vec<U>>(),x1)
}

/// Evaluation of the `Lagrange2dInterpolator` with the data `(x1a,x2a,ya)` 
/// for all `(x1,x2)` given in matching vectors (`assert_eq!(x1.len(),x2.len())`).
/// The data in `ya` should be sorted such that `ya[0]` contains the values 
/// along `x2a` for the first value in `x1a`, and so on.
/// 
/// # Panics
/// 
/// This function will panic if two values in `x1a` or `x2a` are identical.
pub fn lag2_eval_vec<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, ya: &Vec<Vec<U>>, x1: &Vec<T>, x2: &Vec<T>) -> Vec<U>
where 
T: LagRealTrait,
U: LagComplexTrait<T>  {
    // 
    if x1.len() != x2.len() {
        panic!("Input error: x1 and x2 should have the same length.");
    }
    return x1.iter().zip(x2.iter()).map(|(x,y)| lag2_eval(x1a, x2a, ya, x, y)).collect::<Vec<_>>();
}

/// Same as `lag2_eval_vec` but based on the barycentric formula.
pub fn lag2_eval_vec_barycentric<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, w1a: &Vec<T>, w2a: &Vec<T>, ya: &Vec<Vec<U>>, x1: &Vec<T>, x2: &Vec<T>) -> Vec<U>
where 
T: LagRealTrait,
U: LagComplexTrait<T>  {
    // 
    if x1.len() != x2.len() {
        panic!("Input error: x1 and x2 should have the same length.");
    }
    return x1.iter().zip(x2.iter()).map(|(x,y)| lag2_eval_barycentric(x1a, x2a, w1a, w2a, ya, x, y)).collect::<Vec<_>>();
}

/// Evaluation of the `Lagrange2dInterpolator` with the data `(x1a,x2a,ya)` 
/// for computation coordinates given as grid.
/// The data in `ya` should be sorted such that `ya[0]` contains the values 
/// along `x2a` for the first value in `x1a`, and so on.
/// 
/// # Panics
/// 
/// This function will panic if two values in `x1a` or `x2a` are identical.
pub fn lag2_eval_grid<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, ya: &Vec<Vec<U>>, x1: &Vec<T>, x2: &Vec<T>) -> Vec<U>
where 
T: LagRealTrait,
U: LagComplexTrait<T>  {
    // 
    let mut res = vec![zero::<U>();x1.len()*x2.len()];
    for i1 in 0..x1.len() {
        for i2 in 0..x2.len() {
            res[i1*x2.len()+i2] = lag2_eval(x1a,x2a,ya,&x1[i1],&x2[i2]);
        }
    }
    // 
    return res;
}

/// Same as `lag2_eval_grid` but based on the barycentric formula.
pub fn lag2_eval_grid_barycentric<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, w1a: &Vec<T>, w2a: &Vec<T>, ya: &Vec<Vec<U>>, x1: &Vec<T>, x2: &Vec<T>) -> Vec<U>
where 
T: LagRealTrait,
U: LagComplexTrait<T>  {
    // 
    let mut res = vec![zero::<U>();x1.len()*x2.len()];
    for i1 in 0..x1.len() {
        for i2 in 0..x2.len() {
            res[i1*x2.len()+i2] = lag2_eval_barycentric(x1a,x2a,w1a,w2a,ya,&x1[i1],&x2[i2]);
        }
    }
    // 
    return res;
}

/// Evaluation of the first `x1`-derivative of a `Lagrange2dInterpolator` at `(x1,x2)`.
/// 
/// # Panics
/// 
/// This function will panic if two values in `x1a` or `x2a` are identical.
pub fn lag2_diff1<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, ya: &Vec<Vec<U>>, x1: &T, x2: &T) -> U 
where 
T: LagRealTrait,
U: LagComplexTrait<T>  {
    lag1_eval(x1a,&ya.iter().enumerate().map(|(idx,val)| U::from(partial_sum(x1a,x1,idx)).unwrap()*lag1_eval(x2a,val,x2)).collect::<Vec<U>>(),x1)
}

/// Same as `lag2_diff1` but based on the barycentric formula.
pub fn lag2_diff1_barycentric<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, w1a: &Vec<T>, w2a: &Vec<T>, ya: &Vec<Vec<U>>, x1: &T, x2: &T) -> U 
where 
T: LagRealTrait,
U: LagComplexTrait<T>  {
    lag1_eval_barycentric(x1a,w1a,&ya.iter().enumerate().map(|(idx,val)| U::from(partial_sum(x1a,x1,idx)).unwrap()*lag1_eval_barycentric(x2a,w2a,val,x2)).collect::<Vec<U>>(),x1)
}

/// Evaluation of the first x1-derivative of a `Lagrange2dInterpolator` at
/// some grid coordinates `(x1,x2)`.
/// 
/// # Panics
/// 
/// This function will panic if two values in `x1a` or `x2a` are identical.
pub fn lag2_diff1_grid<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, ya: &Vec<Vec<U>>, x1: &Vec<T>, x2: &Vec<T>) -> Vec<U>
where 
T: LagRealTrait,
U: LagComplexTrait<T>  {
    // 
    let mut res = Vec::with_capacity(x1.len()*x2.len());
    for i1 in 0..x1.len() {
        for i2 in 0..x2.len() {
            res.push(lag2_diff1(x1a,x2a,ya,&x1[i1],&x2[i2]));
        }
    }
    // 
    return res;
}

/// Same as `lag2_eval_diff1_grid` but based on the barycentric formula.
pub fn lag2_diff1_grid_barycentric<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, w1a: &Vec<T>, w2a: &Vec<T>, ya: &Vec<Vec<U>>, x1: &Vec<T>, x2: &Vec<T>) -> Vec<U>
where 
T: LagRealTrait,
U: LagComplexTrait<T>  {
    // 
    let mut res = Vec::with_capacity(x1.len()*x2.len());
    for i1 in 0..x1.len() {
        for i2 in 0..x2.len() {
            res.push(lag2_diff1_barycentric(x1a,x2a,w1a,w2a,ya,&x1[i1],&x2[i2]));
        }
    }
    // 
    return res;
}

/// Evaluation of the first x2-derivative of a `Lagrange2dInterpolator` at `(x1,x2)`.
/// 
/// # Panics
/// 
/// This function will panic if two values in `x1a` or `x2a` are identical.
pub fn lag2_diff2<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, ya: &Vec<Vec<U>>, x1: &T, x2: &T) -> U 
where 
T: LagRealTrait,
U: LagComplexTrait<T>  {
    lag1_eval(x1a,&ya.iter().map(|e| {
        lag1_eval(x2a,&e.iter().enumerate().map(|(idx,&val)| val*U::from(partial_sum(x2a,x2,idx)).unwrap()).collect::<Vec<U>>(),x2)
    }).collect::<Vec<U>>(),x1)
}

/// Same as `lag2_eval_diff2` but based on the barycentric formula.
pub fn lag2_diff2_barycentric<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, w1a: &Vec<T>, w2a: &Vec<T>, ya: &Vec<Vec<U>>, x1: &T, x2: &T) -> U 
where 
T: LagRealTrait,
U: LagComplexTrait<T>  {
    lag1_eval_barycentric(x1a,w1a,&ya.iter().map(|e| {
        lag1_eval_barycentric(x2a,w2a,&e.iter().enumerate().map(|(idx,&val)| val*U::from(partial_sum(x2a,x2,idx)).unwrap()).collect::<Vec<U>>(),x2)
    }).collect::<Vec<U>>(),x1)
}

/// Evaluation of the first x2-derivative of a `Lagrange2dInterpolator` at
/// some grid coordinates `(x1,x2)`.
/// 
/// # Panics
/// 
/// This function will panic if two values in `x1a` or `x2a` are identical.
pub fn lag2_diff2_grid<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, ya: &Vec<Vec<U>>, x1: &Vec<T>, x2: &Vec<T>) -> Vec<U>
where 
T: LagRealTrait,
U: LagComplexTrait<T>  {
    // 
    let mut res = Vec::with_capacity(x1.len()*x2.len());
    for i1 in 0..x1.len() {
        for i2 in 0..x2.len() {
            res.push(lag2_diff2(x1a,x2a,ya,&x1[i1],&x2[i2]));
        }
    }
    // 
    return res;
}

/// Same as `lag2_eval_diff2_grid` but based on the barycentric formula.
pub fn lag2_diff2_grid_barycentric<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, w1a: &Vec<T>, w2a: &Vec<T>, ya: &Vec<Vec<U>>, x1: &Vec<T>, x2: &Vec<T>) -> Vec<U>
where 
T: LagRealTrait,
U: LagComplexTrait<T>  {
    // 
    let mut res = Vec::with_capacity(x1.len()*x2.len());
    for i1 in 0..x1.len() {
        for i2 in 0..x2.len() {
            res.push(lag2_diff2_barycentric(x1a,x2a,w1a,w2a,ya,&x1[i1],&x2[i2]));
        }
    }
    // 
    return res;
}