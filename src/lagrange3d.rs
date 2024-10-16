extern crate num_traits;

pub mod lag3_utilities;

use num_traits::{zero,AsPrimitive};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::fmt::{Debug,Display,Formatter,Result};
use std::ops::{DivAssign, MulAssign};

use lag3_utilities::*;

use super::utilities::*;

#[derive(Debug,Clone)]
pub struct Lagrange3dInterpolator<T,U> {
    x1a: Vec<T>,
    x2a: Vec<T>,
    x3a: Vec<T>,
    ya: Vec<Vec<Vec<U>>>,
    diff1_order: usize,
    diff2_order: usize,
    diff3_order: usize
}

#[derive(Debug,Clone)]
pub struct Lagrange3dInterpolatorVec<T,U> {
    lag3_interps: Vec<Lagrange3dInterpolator<T,U>>
}

impl<T,U> Lagrange3dInterpolatorVec<T,U> where
T: LagRealTrait, i32: AsPrimitive<T>, U: LagComplexTrait + DivAssign<T> + MulAssign<T> {
    // 
    pub fn new(x1a: Vec<Vec<T>>, x2a: Vec<Vec<T>>, x3a: Vec<Vec<T>>, ya: Vec<Vec<U>>) -> Lagrange3dInterpolatorVec<T,U> {
        if x1a.len() != x2a.len() || x1a.len() != x3a.len() || x1a.len() != ya.len() || x3a.len() != ya.len() {
            panic!("Error initializing the vector-field interpolator: inputs sizes do not match");
        }
        return Lagrange3dInterpolatorVec { 
            lag3_interps: ya.iter().zip(x1a.iter()).zip(x2a.iter()).zip(x3a.iter()).map(|(((y,x1),x2), x3)| Lagrange3dInterpolator::new((*x1).clone(), (*x2).clone(), (*x3).clone(), (*y).clone())).collect::<Vec<_>>()
         };
    }

    pub fn eval(&self, x1: &T, x2: &T, x3: &T) -> Vec<U> {
        return self.lag3_interps.iter().map(|interp| interp.eval(x1, x2, x3)).collect::<Vec<_>>();
    }
    
    pub fn eval_grid(&self, x1: &Vec<T>, x2: &Vec<T>, x3: &Vec<T>) -> Vec<Vec<U>>{
        return self.lag3_interps.iter().map(|interp| interp.eval_grid(x1, x2, x3)).collect::<Vec<_>>();
    }

    pub fn eval_vec(&self, x1: &Vec<T>, x2: &Vec<T>, x3: &Vec<T>) -> Vec<Vec<U>> {
        return self.lag3_interps.iter().map(|interp| interp.eval_vec(x1, x2, x3)).collect::<Vec<_>>();
    }
    
    pub fn eval_arr(&self, x: &Vec<[T;3]>) -> Vec<Vec<U>> {
        return self.lag3_interps.iter().map(|interp| interp.eval_arr(x)).collect::<Vec<_>>();
    }

    pub fn eval_tup(&self, x: &Vec<(T,T,T)>) -> Vec<Vec<U>> {
        return self.lag3_interps.iter().map(|interp| interp.eval_tup(x)).collect::<Vec<_>>();
    }

    pub fn par_eval_grid(&self, x1: &Vec<T>, x2: &Vec<T>, x3: &Vec<T>) -> Vec<Vec<U>>{
        return self.lag3_interps.iter().map(|interp| interp.par_eval_grid(x1, x2,x3)).collect::<Vec<_>>();
    }

    pub fn par_eval_vec(&self, x1: &Vec<T>, x2: &Vec<T>, x3: &Vec<T>) -> Vec<Vec<U>> {
        return self.lag3_interps.iter().map(|interp| interp.par_eval_vec(x1, x2,x3)).collect::<Vec<_>>();
    }
    
    pub fn par_eval_arr(&self, x: &Vec<[T;3]>) -> Vec<Vec<U>> {
        return self.lag3_interps.iter().map(|interp| interp.par_eval_arr(x)).collect::<Vec<_>>();
    }

    pub fn par_eval_tup(&self, x: &Vec<(T,T,T)>) -> Vec<Vec<U>> {
        return self.lag3_interps.iter().map(|interp| interp.par_eval_tup(x)).collect::<Vec<_>>();
    }

    pub fn jacobian(&self) -> Vec<[Lagrange3dInterpolator<T,U>;3]> {
        self.lag3_interps.iter().map(|interp| [interp.differentiate_x1(),interp.differentiate_x2(),interp.differentiate_x3()]).collect::<Vec<_>>()
    }
    
    pub fn get_inner_interpolators(&self) -> Vec<Lagrange3dInterpolator<T,U>> {
        return self.lag3_interps.clone();
    }

    pub fn order(&self) -> Vec<(usize,usize,usize)> {
        self.lag3_interps.iter().map(|interp| interp.order()).collect::<Vec<_>>()
    }
    
    pub fn len(&self) -> Vec<(usize,usize,usize)> {
        self.lag3_interps.iter().map(|interp| interp.len()).collect::<Vec<_>>()
    }

    pub fn dim(&self) -> usize {
        return self.lag3_interps.len();
    }
}

impl<T,U> Lagrange3dInterpolator<T,U> where
T: LagRealTrait, i32: AsPrimitive<T>, U: LagComplexTrait + DivAssign<T> + MulAssign<T> {
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
            ya: ya,
            diff1_order: 0,
            diff2_order: 0,
            diff3_order: 0
        };
    }

    pub fn order(&self) -> (usize,usize,usize) {
        (self.x1a.len()-1,self.x2a.len()-1,self.x3a.len()-1)
    }
    
    pub fn len(&self) -> (usize,usize,usize) {
        (self.x1a.len(),self.x2a.len(),self.x3a.len())
    }
    
    pub fn diff_orderorder(&self) -> (usize,usize,usize) {
        (self.diff1_order,self.diff2_order,self.diff3_order)
    }

    pub fn get_interp_data(&self) -> (Vec<T>,Vec<T>,Vec<T>,Vec<Vec<Vec<U>>>) {
        (self.x1a.clone(),self.x2a.clone(),self.x3a.clone(),self.ya.clone())
    }

    pub fn get_interp_data_ref(&self) -> (&Vec<T>,&Vec<T>,&Vec<T>,&Vec<Vec<Vec<U>>>) {
        (&(self.x1a),&(self.x2a),&(self.x3a),&(self.ya))
    }

    pub fn eval(&self, x1: &T, x2: &T, x3: &T) -> U {
        lag3_eval(&self.x1a, &self.x2a, &self.x3a, &self.ya, x1, x2, x3)
    }

    pub fn eval_grid(&self, x1: &Vec<T>, x2: &Vec<T>, x3: &Vec<T>) -> Vec<U> {
        lag3_eval_grid(&self.x1a, &self.x2a, &self.x3a, &self.ya, x1, x2,x3)
    }

    pub fn eval_vec(&self, x1: &Vec<T>, x2: &Vec<T>, x3: &Vec<T>) -> Vec<U> {
        lag3_eval_vec(&self.x1a, &self.x2a, &self.x3a, &self.ya, x1, x2, x3)
    }
    
    pub fn eval_arr(&self, x: &Vec<[T;3]>) -> Vec<U> {
        x.iter().map(|e| lag3_eval(&self.x1a, &self.x2a, &self.x3a, &self.ya, &e[0], &e[1],&e[2])).collect::<Vec<_>>()
    }

    pub fn eval_tup(&self, x: &Vec<(T,T,T)>) -> Vec<U> {
        x.iter().map(|e| lag3_eval(&self.x1a, &self.x2a, &self.x3a, &self.ya, &e.0, &e.1, &e.2)).collect::<Vec<_>>()
    }

    pub fn par_eval_grid(&self, x1: &Vec<T>, x2: &Vec<T>, x3: &Vec<T>) -> Vec<U> {
        (*x1).par_iter().flat_map_iter(|xx1| (*x2).iter().flat_map(|xx2| (*x3).iter().map(|xx3| self.eval(xx1, xx2, xx3)))).collect::<Vec<_>>()
    }

    pub fn par_eval_vec(&self, x1: &Vec<T>, x2: &Vec<T>, x3: &Vec<T>) -> Vec<U> {
        (*x1).par_iter().zip_eq((*x2).par_iter()).zip_eq((*x3).par_iter()).map(|((xx1,xx2),xx3)| self.eval(xx1, xx2, xx3)).collect::<Vec<_>>()
    }

    pub fn par_eval_arr(&self, x: &Vec<[T;3]>) -> Vec<U> {
        (*x).par_iter().map(|&xx| self.eval(&xx[0], &xx[1], &xx[2])).collect::<Vec<U>>()
    }

    pub fn par_eval_tup(&self, x: &Vec<(T,T,T)>) -> Vec<U> {
        (*x).par_iter().map(|&(x1,x2,x3)| self.eval(&x1,&x2,&x3)).collect::<Vec<_>>()
    }

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
            let ya_new = lag3_diff1_grid(&x1a, &x2a, &x3a, &ya, &x1a_new, &x2a, &x3a);

            let mut output = Lagrange3dInterpolator::new(x1a_new,x2a,x3a,ya_new);
            output.diff1_order = new_diff1_order;
            output.diff2_order = self.diff2_order;
            output.diff3_order = self.diff3_order;
            return output;
        }
    }

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
            let ya_new = lag3_diff2_grid(&x1a, &x2a, &x3a, &ya, &x1a, &x2a_new, &x3a);

            let mut output = Lagrange3dInterpolator::new(x1a,x2a_new,x3a,ya_new);
            output.diff1_order = self.diff1_order;
            output.diff2_order = new_diff2_order;
            output.diff3_order = self.diff3_order;
            return output;
        }
    }

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
            let ya_new = lag3_diff3_grid(&x1a, &x2a, &x3a, &ya, &x1a, &x2a, &x3a_new);

            let mut output = Lagrange3dInterpolator::new(x1a,x2a,x3a_new,ya_new);
            output.diff1_order = self.diff1_order;
            output.diff2_order = self.diff2_order;
            output.diff3_order = new_diff3_order;
            return output;
        }
    }

    pub fn gradient(&self) -> [Lagrange3dInterpolator<T,U>;3] {
        return [self.differentiate_x1(),self.differentiate_x2(),self.differentiate_x3()] ;
    }
}

impl<T: LagRealTrait, U: LagComplexTrait> Display for Lagrange3dInterpolator<T,U> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f,"Lagrange 2d gridded interpolator:\n- length = {} x {} x {}\n- differentiation order = ({},{},{})",self.x1a.len(),self.x2a.len(), self.x3a.len(),self.diff1_order,self.diff2_order,self.diff3_order)
    }
}
// TESTS
#[cfg(test)]
pub mod lag3_tests;