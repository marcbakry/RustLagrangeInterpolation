extern crate num_traits;

use num_traits::{zero,AsPrimitive};
use std::fmt::{Debug,Display,Formatter,Result};

use super::utilities::*;
use super::lag2_utilities::*;

#[derive(Debug,Clone)]
pub struct Lagrange2dInterpolator<T,U> {
    x1a: Vec<T>,
    x2a: Vec<T>,
    ya: Vec<Vec<U>>,
    diff1_order: usize,
    diff2_order: usize
}

impl<T,U> Lagrange2dInterpolator<T,U> where 

T: LagRealTrait,
i32: AsPrimitive<T>,
U: LagComplexTrait {
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
        let x2a = idx1a.iter().map(|&i| x2a[i]).collect::<Vec<T>>();

        let mut ya_sorted = Vec::with_capacity(x1a.len());
        for i1a in 0..x1a.len() {
            let beg = i1a*x2a.len();
            let tmp = idx2a.iter().map(|&idx| ya[idx+beg]).collect::<Vec<U>>();
            ya_sorted.push(tmp);
        }
        let ya = idx1a.into_iter().map(|idx| ya_sorted[idx].clone()).collect::<Vec<_>>();
        // 
        return Lagrange2dInterpolator { x1a: x1a, x2a: x2a, ya: ya, diff1_order: 0, diff2_order: 0 };
    }

    pub fn order(&self) -> (usize,usize) {
        (self.x1a.len()-1,self.x2a.len()-1)
    }
    
    pub fn len(&self) -> (usize,usize) {
        (self.x1a.len(),self.x2a.len())
    }

    pub fn diff_order(&self) -> (usize,usize) {
        (self.diff1_order,self.diff2_order)
    }

    pub fn get_interp_data(&self) -> (Vec<T>,Vec<T>,Vec<Vec<U>>) {
        (self.x1a.clone(),self.x2a.clone(),self.ya.clone())
    }

    pub fn get_interp_data_ref(&self) -> (&Vec<T>,&Vec<T>,&Vec<Vec<U>>) {
        (&(self.x1a),&(self.x2a),&(self.ya))
    }

    pub fn eval(&self,x1: T, x2: T) -> U {
        lag2_eval(&self.x1a, &self.x2a, &self.ya, x1, x2)
    }

    pub fn eval_grid(&self, x1: Vec<T>, x2: Vec<T>) -> Vec<U> {
        lag2_eval_grid(&self.x1a, &self.x2a, &self.ya, x1, x2)
    }

    pub fn differentiate_x1(&self) -> Lagrange2dInterpolator<T, U> {
        // 
        let (x1a,x2a, mut ya) = self.get_interp_data();
        let new_diff1_order = self.diff1_order+1;
        // 
        if self.x1a.len()-1 == 0 {
            ya.iter_mut().for_each(|y| y.iter_mut().for_each(|e| *e = zero::<U>()));

            return Lagrange2dInterpolator {
                x1a: x1a,
                x2a: x2a,
                ya: ya,
                diff1_order: new_diff1_order,
                diff2_order: self.diff2_order
            };
        } else {
            let x1a_new = midpoints(&x1a);
            let ya_new = lag2_diff1_grid(&x1a, &x2a, &ya, &x1a_new, &x2a);
            let mut output = Lagrange2dInterpolator::new(x1a_new,x2a,ya_new);
            output.diff1_order = new_diff1_order;
            output.diff2_order = self.diff2_order;
            return output;
        }
    }
    
    pub fn differentiate_x2(&self) -> Lagrange2dInterpolator<T, U> {
        // 
        let (x1a,x2a, mut ya) = self.get_interp_data();
        let new_diff2_order = self.diff2_order+1;
        // 
        if self.x2a.len()-1 == 0 {
            ya.iter_mut().for_each(|y| y.iter_mut().for_each(|e| *e = zero::<U>()));

            return Lagrange2dInterpolator {
                x1a: x1a,
                x2a: x2a,
                ya: ya,
                diff1_order: self.diff1_order,
                diff2_order: new_diff2_order
            };
        } else {
            let x2a_new = midpoints(&x2a);
            let ya_new = lag2_diff2_grid(&x1a, &x2a, &ya, &x1a, &x2a_new);
            let mut output = Lagrange2dInterpolator::new(x1a,x2a_new,ya_new);
            output.diff1_order = self.diff1_order;
            output.diff2_order = new_diff2_order;
            return output;
        }
    }
}

impl<T: LagRealTrait, U: LagComplexTrait> Display for Lagrange2dInterpolator<T,U> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f,"Lagrange 2d gridded interpolator:\n- length = {} x {}\n- differentiation order = ({},{})",self.x1a.len(),self.x2a.len(),self.diff1_order,self.diff2_order)
    }
}