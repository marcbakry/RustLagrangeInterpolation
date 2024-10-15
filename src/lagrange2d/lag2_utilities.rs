use super::super::utilities::{partial_sum,LagRealTrait,LagComplexTrait}; 
use crate::lagrange1d::lag1_utilities::*;

use num_traits::zero;
use std::ops::{DivAssign, MulAssign};

pub fn lag2_eval<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, ya: &Vec<Vec<U>>, x1: &T, x2: &T) -> U 
where 
T: LagRealTrait,
U: LagComplexTrait + DivAssign<T> + MulAssign<T> {
    lag1_eval(x1a,&ya.iter().map(|e| lag1_eval(x2a,e,x2)).collect::<Vec<U>>(),x1)
}

pub fn lag2_eval_vec<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, ya: &Vec<Vec<U>>, x1: &Vec<T>, x2: &Vec<T>) -> Vec<U>
where 
T: LagRealTrait,
U: LagComplexTrait + DivAssign<T> + MulAssign<T>  {
    // 
    if x1.len() != x2.len() {
        panic!("Input error: x1 and x2 should have the same length.");
    }
    return x1.iter().zip(x2.iter()).map(|(x,y)| lag2_eval(x1a, x2a, ya, x, y)).collect::<Vec<_>>();
}

pub fn lag2_eval_grid<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, ya: &Vec<Vec<U>>, x1: &Vec<T>, x2: &Vec<T>) -> Vec<U>
where 
T: LagRealTrait,
U: LagComplexTrait + DivAssign<T> + MulAssign<T>  {
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

pub fn lag2_diff1<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, ya: &Vec<Vec<U>>, x1: &T, x2: &T) -> U 
where 
T: LagRealTrait,
U: LagComplexTrait + DivAssign<T> + MulAssign<T>  {
    lag1_eval(x1a,&ya.iter().enumerate().map(|(idx,val)| U::from(partial_sum(x1a,x1,idx)).unwrap()*lag1_eval(x2a,val,x2)).collect::<Vec<U>>(),x1)
}

pub fn lag2_diff1_grid<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, ya: &Vec<Vec<U>>, x1: &Vec<T>, x2: &Vec<T>) -> Vec<U>
where 
T: LagRealTrait,
U: LagComplexTrait + DivAssign<T> + MulAssign<T>  {
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

pub fn lag2_diff2<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, ya: &Vec<Vec<U>>, x1: &T, x2: &T) -> U 
where 
T: LagRealTrait,
U: LagComplexTrait + DivAssign<T> + MulAssign<T>  {
    lag1_eval(x1a,&ya.iter().map(|e| {
        lag1_eval(x2a,&e.iter().enumerate().map(|(idx,&val)| val*U::from(partial_sum(x2a,x2,idx)).unwrap()).collect::<Vec<U>>(),x2)
    }).collect::<Vec<U>>(),x1)
}

pub fn lag2_diff2_grid<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, ya: &Vec<Vec<U>>, x1: &Vec<T>, x2: &Vec<T>) -> Vec<U>
where 
T: LagRealTrait,
U: LagComplexTrait + DivAssign<T> + MulAssign<T>  {
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
