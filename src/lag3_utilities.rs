use super::utilities::{partial_sum,LagRealTrait,LagComplexTrait}; 
use super::lag1_utilities::*;
use super::lag2_utilities::*;

use num_traits::zero;
use std::ops::{DivAssign, MulAssign};

pub fn lag3_eval<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, x3a: &Vec<T>, ya: &Vec<Vec<Vec<U>>>, x1: &T, x2: &T, x3: &T) -> U 
where 
T: LagRealTrait,
U: LagComplexTrait + DivAssign<T> + MulAssign<T> {
    lag1_eval(x1a, &ya.iter().map(|e| lag2_eval(x2a,x3a,e,x2,x3)).collect::<Vec<U>>(),x1)
}

pub fn lag3_eval_grid<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, x3a: &Vec<T>, ya: &Vec<Vec<Vec<U>>>, x1: &Vec<T>, x2: &Vec<T>, x3: &Vec<T>) -> Vec<U>
where 
T: LagRealTrait,
U: LagComplexTrait + DivAssign<T> + MulAssign<T>  {
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

pub fn lag3_eval_vec<T,U>(x1a: &Vec<T>, x2a: &Vec<T>, x3a: &Vec<T>, ya: &Vec<Vec<Vec<U>>>, x1: &Vec<T>, x2: &Vec<T>, x3: &Vec<T>) -> Vec<U>
where 
T: LagRealTrait,
U: LagComplexTrait + DivAssign<T> + MulAssign<T>  {
    // 
    if x1.len() != x2.len() || x1.len() != x3.len() {
        panic!("Input error: x1 and x2 and x3 should have the same length.");
    }
    return x1.iter().zip(x2.iter()).zip(x3.iter()).map(|((x,y),z)| lag3_eval(x1a, x2a, x3a, ya, x, y,z)).collect::<Vec<_>>();
}