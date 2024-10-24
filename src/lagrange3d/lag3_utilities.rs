//! Module containing the low-level functions for the construction and the evaluation
//! of the `Lagrange3dInterpolator`.
use super::super::utilities::{partial_sum,LagRealTrait,LagComplexTrait}; 
use crate::lagrange1d::lag1_utilities::*;
use crate::lagrange2d::lag2_utilities::*;

use num_traits::zero;


/// Evaluation of the `Lagrange3dInterpolator` with the data `(x1a,x2a,x3a,ya)` at `(x1,x2,x3)`.
/// The data in `ya` should be sorted such that `ya[0]` contains the values 
/// along `x2a` for the first value in `x1a`, and so on.
/// 
/// # Panics
/// 
/// This function will panic if two values in `x1a` or `x2a` are identical.
pub fn lag3_eval<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, x3a: &Vec<T>, ya: &Vec<Vec<Vec<U>>>, x1: &T, x2: &T, x3: &T) -> U 
where 
T: LagRealTrait,
U: LagComplexTrait<T> {
    lag1_eval(x1a, &ya.iter().map(|e| lag2_eval(x2a,x3a,e,x2,x3)).collect::<Vec<U>>(),x1)
}

pub fn lag3_eval_barycentric<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, x3a: &Vec<T>, w1a: &Vec<T>, w2a: &Vec<T>, w3a: &Vec<T>, ya: &Vec<Vec<Vec<U>>>, x1: &T, x2: &T, x3: &T) -> U 
where 
T: LagRealTrait,
U: LagComplexTrait<T> {
    lag1_eval_barycentric(x1a,w1a,&ya.iter().map(|e| lag2_eval_barycentric(x2a,x3a,w2a,w3a,e,x2,x3)).collect::<Vec<U>>(),x1)
}

/// Evaluation of the `Lagrange3dInterpolator` with the data `(x1a,x2a,x3a,ya)` 
/// for computation coordinates given as grid.
/// The data in `ya` should be sorted such that `ya[0]` contains the values 
/// along `x2a` for the first value in `x1a`, and so on.
/// 
/// # Panics
/// 
/// This function will panic if two values in `x1a` or `x2a` or `x3a` are identical.
pub fn lag3_eval_grid<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, x3a: &Vec<T>, ya: &Vec<Vec<Vec<U>>>, x1: &Vec<T>, x2: &Vec<T>, x3: &Vec<T>) -> Vec<U>
where 
T: LagRealTrait,
U: LagComplexTrait<T>  {
    // 
    let mut res = vec![zero::<U>();x1.len()*x2.len()*x3.len()];
    for i1 in 0..x1.len() {
        for i2 in 0..x2.len() {
            for i3 in 0..x3.len() {
                res[i1*x2.len()*x3.len()+i2*x3.len() + i3] = lag3_eval(x1a,x2a,x3a,ya,&x1[i1],&x2[i2],&x3[i3]);
            }
        }
    }
    // 
    return res;
}

pub fn lag3_eval_grid_barycentric<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, x3a: &Vec<T>, w1a: &Vec<T>, w2a: &Vec<T>, w3a: &Vec<T>, ya: &Vec<Vec<Vec<U>>>, x1: &Vec<T>, x2: &Vec<T>, x3: &Vec<T>) -> Vec<U>
where 
T: LagRealTrait,
U: LagComplexTrait<T>  {
    // 
    let mut res = vec![zero::<U>();x1.len()*x2.len()*x3.len()];
    for i1 in 0..x1.len() {
        for i2 in 0..x2.len() {
            for i3 in 0..x3.len() {
                res[i1*x2.len()*x3.len()+i2*x3.len() + i3] = lag3_eval_barycentric(x1a,x2a,x3a,w1a,w2a,w3a,ya,&x1[i1],&x2[i2],&x3[i3]);
            }
        }
    }
    // 
    return res;
}

/// Evaluation of the `Lagrange3dInterpolator` with the data `(x1a,x2a,x3a,ya)` 
/// for all `(x1,x2,x3)` given in matching vectors (`assert_eq!(x1.len(),x2.len())`) etc.
/// The data in `ya` should be sorted such that `ya[0]` contains the values 
/// along `x2a` for the first value in `x1a`, and so on.
/// 
/// # Panics
/// 
/// This function will panic if two values in `x1a` or `x2a` or `x3a` are identical.
pub fn lag3_eval_vec<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, x3a: &Vec<T>, ya: &Vec<Vec<Vec<U>>>, x1: &Vec<T>, x2: &Vec<T>, x3: &Vec<T>) -> Vec<U>
where 
T: LagRealTrait,
U: LagComplexTrait<T>  {
    // 
    if x1.len() != x2.len() || x1.len() != x3.len() {
        panic!("Input error: x1 and x2 and x3 should have the same length.");
    }
    return x1.iter().zip(x2.iter()).zip(x3.iter()).map(|((x,y),z)| lag3_eval(x1a, x2a, x3a, ya, x, y,z)).collect::<Vec<_>>();
}

pub fn lag3_eval_vec_barycentric<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, x3a: &Vec<T>, w1a: &Vec<T>, w2a: &Vec<T>, w3a: &Vec<T>, ya: &Vec<Vec<Vec<U>>>, x1: &Vec<T>, x2: &Vec<T>, x3: &Vec<T>) -> Vec<U>
where 
T: LagRealTrait,
U: LagComplexTrait<T>  {
    // 
    if x1.len() != x2.len() || x1.len() != x3.len() {
        panic!("Input error: x1 and x2 and x3 should have the same length.");
    }
    return x1.iter().zip(x2.iter()).zip(x3.iter()).map(|((x,y),z)| lag3_eval_barycentric(x1a, x2a, x3a, w1a, w2a, w3a, ya, x, y,z)).collect::<Vec<_>>();
}

/// Evaluation of the first x1-derivative of a `Lagrange3dInterpolator` at `(x1,x2,x3)`.
/// 
/// # Panics
/// 
/// This function will panic if two values in `x1a` or `x2a` or `x3a` are identical.
pub fn lag3_diff1<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, x3a: &Vec<T>, ya: &Vec<Vec<Vec<U>>>, x1: &T, x2: &T, x3: &T) -> U 
where 
T: LagRealTrait,
U: LagComplexTrait<T> {    
    lag1_eval(x1a, 
        &ya.iter().enumerate().map(
        |(idx,val)| U::from(partial_sum(x1a, x1, idx)).unwrap()*lag2_eval(x2a, x3a, val, x2, x3)
    ).collect::<Vec<U>>(), 
    x1)
}

pub fn lag3_diff1_barycentric<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, x3a: &Vec<T>, w1a: &Vec<T>, w2a: &Vec<T>, w3a: &Vec<T>, ya: &Vec<Vec<Vec<U>>>, x1: &T, x2: &T, x3: &T) -> U 
where 
T: LagRealTrait,
U: LagComplexTrait<T> {    
    lag1_eval_barycentric(x1a, 
        w1a,
        &ya.iter().enumerate().map(
        |(idx,val)| U::from(partial_sum(x1a, x1, idx)).unwrap()*lag2_eval_barycentric(x2a, x3a, w2a, w3a, val, x2, x3)
    ).collect::<Vec<U>>(), 
    x1)
}

/// Evaluation of the first x1-derivative of a `Lagrange3dInterpolator` at
/// some grid coordinates `(x1,x2,x3)`.
/// 
/// # Panics
/// 
/// This function will panic if two values in `x1a` or `x2a` are identical.
pub fn lag3_diff1_grid<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, x3a: &Vec<T>, ya: &Vec<Vec<Vec<U>>>, x1: &Vec<T>, x2: &Vec<T>, x3: &Vec<T>) -> Vec<U>
where 
T: LagRealTrait,
U: LagComplexTrait<T>  {
    // 
    let mut res = vec![zero::<U>();x1.len()*x2.len()*x3.len()];
    for i1 in 0..x1.len() {
        for i2 in 0..x2.len() {
            for i3 in 0..x3.len() {
                res[i1*x2.len()*x3.len()+i2*x3.len() + i3] = lag3_diff1(x1a,x2a,x3a,ya,&x1[i1],&x2[i2],&x3[i3]);
            }
        }
    }
    // 
    return res;
}

pub fn lag3_diff1_grid_barycentric<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, x3a: &Vec<T>, w1a: &Vec<T>, w2a: &Vec<T>, w3a: &Vec<T>, ya: &Vec<Vec<Vec<U>>>, x1: &Vec<T>, x2: &Vec<T>, x3: &Vec<T>) -> Vec<U>
where 
T: LagRealTrait,
U: LagComplexTrait<T>  {
    // 
    let mut res = vec![zero::<U>();x1.len()*x2.len()*x3.len()];
    for i1 in 0..x1.len() {
        for i2 in 0..x2.len() {
            for i3 in 0..x3.len() {
                res[i1*x2.len()*x3.len()+i2*x3.len() + i3] = lag3_diff1_barycentric(x1a,x2a,x3a,w1a,w2a,w3a,ya,&x1[i1],&x2[i2],&x3[i3]);
            }
        }
    }
    // 
    return res;
}

/// Evaluation of the first x2-derivative of a `Lagrange3dInterpolator` at `(x1,x2,x3)`.
/// 
/// # Panics
/// 
/// This function will panic if two values in `x1a` or `x2a` or `x3a` are identical.
pub fn lag3_diff2<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, x3a: &Vec<T>, ya: &Vec<Vec<Vec<U>>>, x1: &T, x2: &T, x3: &T) -> U 
where 
T: LagRealTrait,
U: LagComplexTrait<T> {  
    lag1_eval(x1a, &ya.iter().map(|e| lag2_diff1(x2a, x3a, e, x2, x3)).collect::<Vec<U>>(), x1)
}

pub fn lag3_diff2_barycentric<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, x3a: &Vec<T>, w1a: &Vec<T>, w2a: &Vec<T>, w3a: &Vec<T>, ya: &Vec<Vec<Vec<U>>>, x1: &T, x2: &T, x3: &T) -> U 
where 
T: LagRealTrait,
U: LagComplexTrait<T> {  
    lag1_eval_barycentric(x1a, w1a, &ya.iter().map(|e| lag2_diff1_barycentric(x2a, x3a, w2a, w3a, e, x2, x3)).collect::<Vec<U>>(), x1)
}

/// Evaluation of the first x2-derivative of a `Lagrange3dInterpolator` at
/// some grid coordinates `(x1,x2,x3)`.
/// 
/// # Panics
/// 
/// This function will panic if two values in `x1a` or `x2a` are identical.
pub fn lag3_diff2_grid<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, x3a: &Vec<T>, ya: &Vec<Vec<Vec<U>>>, x1: &Vec<T>, x2: &Vec<T>, x3: &Vec<T>) -> Vec<U>
where 
T: LagRealTrait,
U: LagComplexTrait<T>  {
    // 
    let mut res = vec![zero::<U>();x1.len()*x2.len()*x3.len()];
    for i1 in 0..x1.len() {
        for i2 in 0..x2.len() {
            for i3 in 0..x3.len() {
                res[i1*x2.len()*x3.len()+i2*x3.len() + i3] = lag3_diff2(x1a,x2a,x3a,ya,&x1[i1],&x2[i2],&x3[i3]);
            }
        }
    }
    // 
    return res;
}

pub fn lag3_diff2_grid_barycentric<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, x3a: &Vec<T>, w1a: &Vec<T>, w2a: &Vec<T>, w3a: &Vec<T>, ya: &Vec<Vec<Vec<U>>>, x1: &Vec<T>, x2: &Vec<T>, x3: &Vec<T>) -> Vec<U>
where 
T: LagRealTrait,
U: LagComplexTrait<T>  {
    // 
    let mut res = vec![zero::<U>();x1.len()*x2.len()*x3.len()];
    for i1 in 0..x1.len() {
        for i2 in 0..x2.len() {
            for i3 in 0..x3.len() {
                res[i1*x2.len()*x3.len()+i2*x3.len() + i3] = lag3_diff2_barycentric(x1a,x2a,x3a,w1a,w2a,w3a,ya,&x1[i1],&x2[i2],&x3[i3]);
            }
        }
    }
    // 
    return res;
}

/// Evaluation of the first x3-derivative of a `Lagrange3dInterpolator` at `(x1,x2,x3)`.
/// 
/// # Panics
/// 
/// This function will panic if two values in `x1a` or `x2a` or `x3a` are identical.
pub fn lag3_diff3<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, x3a: &Vec<T>, ya: &Vec<Vec<Vec<U>>>, x1: &T, x2: &T, x3: &T) -> U 
where 
T: LagRealTrait,
U: LagComplexTrait<T> {  
    lag1_eval(x1a, &ya.iter().map(|e| lag2_diff2(x2a, x3a, e, x2, x3)).collect::<Vec<U>>(), x1)
}

pub fn lag3_diff3_barycentric<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, x3a: &Vec<T>, w1a: &Vec<T>, w2a: &Vec<T>, w3a: &Vec<T>, ya: &Vec<Vec<Vec<U>>>, x1: &T, x2: &T, x3: &T) -> U 
where 
T: LagRealTrait,
U: LagComplexTrait<T> {  
    lag1_eval_barycentric(x1a, w1a, &ya.iter().map(|e| lag2_diff2_barycentric(x2a, x3a, w2a, w3a, e, x2, x3)).collect::<Vec<U>>(), x1)
}

/// Evaluation of the first x3-derivative of a `Lagrange3dInterpolator` at
/// some grid coordinates `(x1,x2,x3)`.
/// 
/// # Panics
/// 
/// This function will panic if two values in `x1a` or `x2a` are identical.
pub fn lag3_diff3_grid<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, x3a: &Vec<T>, ya: &Vec<Vec<Vec<U>>>, x1: &Vec<T>, x2: &Vec<T>, x3: &Vec<T>) -> Vec<U>
where 
T: LagRealTrait,
U: LagComplexTrait<T>  {
    // 
    let mut res = vec![zero::<U>();x1.len()*x2.len()*x3.len()];
    for i1 in 0..x1.len() {
        for i2 in 0..x2.len() {
            for i3 in 0..x3.len() {
                res[i1*x2.len()*x3.len()+i2*x3.len() + i3] = lag3_diff3(x1a,x2a,x3a,ya,&x1[i1],&x2[i2],&x3[i3]);
            }
        }
    }
    // 
    return res;
}

pub fn lag3_diff3_grid_barycentric<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, x3a: &Vec<T>, w1a: &Vec<T>, w2a: &Vec<T>, w3a: &Vec<T>, ya: &Vec<Vec<Vec<U>>>, x1: &Vec<T>, x2: &Vec<T>, x3: &Vec<T>) -> Vec<U>
where 
T: LagRealTrait,
U: LagComplexTrait<T>  {
    // 
    let mut res = vec![zero::<U>();x1.len()*x2.len()*x3.len()];
    for i1 in 0..x1.len() {
        for i2 in 0..x2.len() {
            for i3 in 0..x3.len() {
                res[i1*x2.len()*x3.len()+i2*x3.len() + i3] = lag3_diff3_barycentric(x1a,x2a,x3a,w1a,w2a,w3a,ya,&x1[i1],&x2[i2],&x3[i3]);
            }
        }
    }
    // 
    return res;
}