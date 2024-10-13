extern crate num_traits;

use num_traits::{zero,AsPrimitive};
use std::fmt::{Debug,Display,Formatter,Result};
use std::ops::{DivAssign, MulAssign};

use crate::lag3_utilities::*;

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
}


// TESTS
#[cfg(test)]
mod lagrange3d_tests {
    use crate::lagrange3d::Lagrange3dInterpolator;

    use super::*;
    use std::f64::consts::PI;

    #[test]
    pub fn lag3_interpolation() {
        let f = |x1: f64, x2: f64, x3: f64| f64::cos(2.0*PI*x1.powi(2))*f64::sin(2.0*PI*x2).powi(2)*f64::tan(x3-0.5);
        let df_x1 = |x1: f64, x2: f64, x3: f64| -4.0*PI*x1*f64::sin(2.0*PI*x1.powi(2))*f64::sin(2.0*PI*x2).powi(2)*f64::tan(x3-0.5);
        let df_x2 = |x1: f64, x2: f64, x3: f64| f64::cos(2.0*PI*x1.powi(2))*4.0*PI*f64::sin(2.0*PI*x2)*f64::cos(2.0*PI*x2)*f64::tan(x3-0.5);
        let df_x3 = |x1: f64, x2: f64, x3: f64| f64::cos(2.0*PI*x1.powi(2))*f64::sin(2.0*PI*x2).powi(2)*(1.0 + f64::tan(x3-0.5).powi(2));
        // 
        let (n1a,n2a,n3a) = (16, 17, 18);
        let (a1,b1) = (0.0,1.0);
        let (a2,b2) = (0.0,1.0);
        let (a3,b3) = (0.0,1.0);
        // let (stp1a,stp2a,stp3a) = ((b1-a1)/((n1a-1) as f64),(b2-a2)/((n2a-1) as f64),(b3-a3)/((n3a-1) as f64));
        let x1a = gauss_chebyshev_nodes(&n1a, &a1, &b1);
        let x2a = gauss_chebyshev_nodes(&n2a, &a2, &b2);
        let x3a = gauss_chebyshev_nodes(&n3a, &a3, &b3);
        let mut ya = Vec::with_capacity(x1a.len()*x2a.len()*x3a.len());
        x1a.iter().for_each(|&x1| x2a.iter().for_each(|&x2| x3a.iter().for_each(|&x3| ya.push(f(x1,x2,x3)))));

        let lag3_f = Lagrange3dInterpolator::new(x1a, x2a, x3a, ya);
        let lag3_df_dx1 = lag3_f.differentiate_x1();
        let lag3_df_dx2 = lag3_f.differentiate_x2();
        let lag3_df_dx3 = lag3_f.differentiate_x3();

        // interpolated data
        let (n1i,n2i,n3i) = (39,40,41);
        let (stp1i,stp2i,stp3i) = ((b1-a1)/((n1i-1) as f64),(b2-a2)/((n2i-1) as f64),(b3-a3)/((n3i-1) as f64));
        let x1i = (0..n1i).map(|i| i as f64*stp1i).collect::<Vec<_>>();
        let x2i = (0..n2i).map(|i| i as f64*stp2i).collect::<Vec<_>>();
        let x3i = (0..n3i).map(|i| i as f64*stp3i).collect::<Vec<_>>();

        let yi_f = lag3_f.eval_grid(&x1i, &x2i, &x3i);
        let yi_df_dx1 = lag3_df_dx1.eval_grid(&x1i, &x2i, &x3i);
        let yi_df_dx2 = lag3_df_dx2.eval_grid(&x1i, &x2i, &x3i);
        let yi_df_dx3 = lag3_df_dx3.eval_grid(&x1i, &x2i, &x3i);

        // reference data
        let mut yref_f = Vec::with_capacity(n1i*n2i*n3i);
        x1i.iter().for_each(|&x1| x2i.iter().for_each(|&x2| x3i.iter().for_each(|&x3| yref_f.push(f(x1,x2,x3)))));
        
        let mut yref_df_dx1 = Vec::with_capacity(n1i*n2i*n3i);
        x1i.iter().for_each(|&x1| x2i.iter().for_each(|&x2| x3i.iter().for_each(|&x3| yref_df_dx1.push(df_x1(x1,x2,x3)))));
        
        let mut yref_df_dx2 = Vec::with_capacity(n1i*n2i*n3i);
        x1i.iter().for_each(|&x1| x2i.iter().for_each(|&x2| x3i.iter().for_each(|&x3| yref_df_dx2.push(df_x2(x1,x2,x3)))));
        
        let mut yref_df_dx3 = Vec::with_capacity(n1i*n2i*n3i);
        x1i.iter().for_each(|&x1| x2i.iter().for_each(|&x2| x3i.iter().for_each(|&x3| yref_df_dx3.push(df_x3(x1,x2,x3)))));

        // check accuracy with the maximum of the absolute error
        let err_f = yi_f.iter().zip(yref_f.iter()).map(|(ei,ef)| (ei-ef).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap();
        let err_df_dx1 = yi_df_dx1.iter().zip(yref_df_dx1.iter()).map(|(ei,ef)| (ei-ef).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap();
        let err_df_dx2 = yi_df_dx2.iter().zip(yref_df_dx2.iter()).map(|(ei,ef)| (ei-ef).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap();
        let err_df_dx3 = yi_df_dx3.iter().zip(yref_df_dx3.iter()).map(|(ei,ef)| (ei-ef).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap();
        
        println!("Error 0-th order derivative         = {}",err_f);
        println!("Error 1-st order derivative (d/dx1) = {}",err_df_dx1);
        println!("Error 1-st order derivative (d/dx2) = {}",err_df_dx2);
        println!("Error 1-st order derivative (d/dx3) = {}",err_df_dx3);

        assert!(err_f < 1e-4);
        assert!(err_df_dx1 < 1e-3);
        assert!(err_df_dx2 < 1e-3);
        assert!(err_df_dx3 < 1e-3);
    }
}