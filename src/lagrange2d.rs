extern crate num_traits;

pub mod lag2_utilities;

use num_traits::{zero,AsPrimitive};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::fmt::{Debug,Display,Formatter,Result};
use std::ops::{Div,DivAssign,Mul,MulAssign,Add,AddAssign,Sub,SubAssign};

use super::utilities::*;
use lag2_utilities::*;

#[derive(Debug,Clone)]
pub struct Lagrange2dInterpolator<T,U> {
    x1a: Vec<T>,
    x2a: Vec<T>,
    ya: Vec<Vec<U>>,
    diff1_order: usize,
    diff2_order: usize
}

#[derive(Debug,Clone)]
pub struct Lagrange2dInterpolatorVec<T,U> {
    lag2_interps: Vec<Lagrange2dInterpolator<T,U>>
}

impl<T,U> Lagrange2dInterpolatorVec<T,U> where
T: LagRealTrait, i32: AsPrimitive<T>, U: LagComplexTrait + DivAssign<T> + MulAssign<T> {
    pub fn new(x1a: Vec<Vec<T>>, x2a: Vec<Vec<T>>, ya: Vec<Vec<U>>) -> Lagrange2dInterpolatorVec<T,U> {
        if x1a.len() != x2a.len() || x1a.len() != ya.len() {
            panic!("Error initializing the vector-field interpolator: inputs sizes do not match");
        }
        return Lagrange2dInterpolatorVec { 
            lag2_interps: ya.iter().zip(x1a.iter()).zip(x2a.iter()).map(|((y,x1),x2)| Lagrange2dInterpolator::new((*x1).clone(), (*x2).clone(), (*y).clone())).collect::<Vec<_>>()
         };
    }

    pub fn eval(&self, x1: &T, x2: &T) -> Vec<U> {
        return self.lag2_interps.iter().map(|interp| interp.eval(x1, x2)).collect::<Vec<_>>();
    }

    pub fn eval_grid(&self, x1: &Vec<T>, x2: &Vec<T>) -> Vec<Vec<U>>{
        return self.lag2_interps.iter().map(|interp| interp.eval_grid(x1, x2)).collect::<Vec<_>>();
    }

    pub fn jacobian(&self) -> Vec<[Lagrange2dInterpolator<T,U>;2]> {
        self.lag2_interps.iter().map(|interp| [interp.differentiate_x1(),interp.differentiate_x2()]).collect::<Vec<_>>()
    }

    pub fn get_inner_interpolators(&self) -> Vec<Lagrange2dInterpolator<T,U>> {
        return self.lag2_interps.clone();
    }

    pub fn order(&self) -> Vec<(usize,usize)> {
        self.lag2_interps.iter().map(|interp| interp.order()).collect::<Vec<_>>()
    }
    
    pub fn len(&self) -> Vec<(usize,usize)> {
        self.lag2_interps.iter().map(|interp| interp.len()).collect::<Vec<_>>()
    }

    pub fn dim(&self) -> usize {
        return self.lag2_interps.len();
    }
}

impl<T,U> Lagrange2dInterpolator<T,U> where 
T: LagRealTrait,
i32: AsPrimitive<T>,
U: LagComplexTrait + DivAssign<T> + MulAssign<T> {
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
        let x2a = idx2a.iter().map(|&i| x2a[i]).collect::<Vec<T>>();

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

    pub fn eval(&self,x1: &T, x2: &T) -> U {
        lag2_eval(&self.x1a, &self.x2a, &self.ya, x1, x2)
    }

    pub fn eval_grid(&self, x1: &Vec<T>, x2: &Vec<T>) -> Vec<U> {
        lag2_eval_grid(&self.x1a, &self.x2a, &self.ya, x1, x2)
    }

    pub fn eval_vec(&self, x1: &Vec<T>, x2: &Vec<T>) -> Vec<U> {
        lag2_eval_vec(&self.x1a, &self.x2a, &self.ya, x1, x2)
    }
    pub fn eval_arr(&self, x: &Vec<[T;2]>) -> Vec<U> {
        x.iter().map(|e| lag2_eval(&self.x1a, &self.x2a, &self.ya, &e[0], &e[1])).collect::<Vec<_>>()
    }

    pub fn eval_tup(&self, x: &Vec<(T,T)>) -> Vec<U> {
        x.iter().map(|e| lag2_eval(&self.x1a, &self.x2a, &self.ya, &e.0, &e.1)).collect::<Vec<_>>()
    }

    pub fn par_eval_grid(&self, x1: &Vec<T>, x2: &Vec<T>) -> Vec<U> {
        (*x1).par_iter().flat_map_iter(|xx1| (*x2).iter().map(|xx2| self.eval(xx1, xx2))).collect::<Vec<_>>()
    }

    pub fn par_eval_vec(&self, x1: &Vec<T>, x2: &Vec<T>) -> Vec<U> {
        (*x1).par_iter().zip_eq((*x2).par_iter()).map(|(xx1,xx2)| self.eval(xx1,xx2)).collect::<Vec<U>>()
    }

    pub fn par_eval_arr(&self, x: &Vec<[T;2]>) -> Vec<U> {
        (*x).par_iter().map(|&xx| self.eval(&xx[0], &xx[1])).collect::<Vec<U>>()
    }

    pub fn par_eval_tup(&self, x: &Vec<(T,T)>) -> Vec<U> {
        (*x).par_iter().map(|&(x1,x2)| self.eval(&x1,&x2)).collect::<Vec<_>>()
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

    pub fn gradient(&self) -> [Lagrange2dInterpolator<T,U>;2] {
        return [self.differentiate_x1(),self.differentiate_x2()];
    }
}

impl<T: LagRealTrait, U: LagComplexTrait> Display for Lagrange2dInterpolator<T,U> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f,"Lagrange 2d gridded interpolator:\n- length = {} x {}\n- differentiation order = ({},{})",self.x1a.len(),self.x2a.len(),self.diff1_order,self.diff2_order)
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> Add<U> for Lagrange2dInterpolator<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange2dInterpolator<T,U>;
    fn add(self, rhs: U) -> Self::Output {
        let (x1a,x2a,ya) = self.get_interp_data();
        let new_ya = ya.iter().flat_map(|y| y.iter()).map(|&e| e-rhs).collect::<Vec<_>>();
        return Lagrange2dInterpolator::new(x1a, x2a, new_ya);
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> Add<Lagrange2dInterpolator<T,U>> for Lagrange2dInterpolator<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange2dInterpolator<T,U>;
    fn add(self, rhs: Lagrange2dInterpolator<T,U>) -> Self::Output {
        let (x1a_lhs,x2a_lhs,ya_lhs) = self.get_interp_data();
        let (x1a_rhs,x2a_rhs,ya_rhs) = rhs.get_interp_data();
        
        let is_same_x1a = x1a_lhs.iter().zip(x1a_rhs.iter()).any(|(&a,&b)| (a-b).abs() > T::from(1e-12).unwrap()) && (x1a_lhs.len() == x1a_rhs.len());
        let is_same_x2a = x2a_lhs.iter().zip(x2a_rhs.iter()).any(|(&a,&b)| (a-b).abs() > T::from(1e-12).unwrap()) && (x2a_lhs.len() == x1a_rhs.len());

        // flatten y
        let ya_lhs = ya_lhs.iter().flat_map(|e| e.iter()).map(|&e| e).collect::<Vec<_>>();
        let ya_rhs = ya_rhs.iter().flat_map(|e| e.iter()).map(|&e| e).collect::<Vec<_>>();

        let (x1a_new,x2a_new,ya_new) = if is_same_x1a && is_same_x2a {
            (x1a_lhs,x2a_lhs,ya_lhs.iter().zip(ya_rhs.iter()).map(|(&a,&b)| a+b).collect::<Vec<_>>())
        } else {
            let x1a_new = if x1a_lhs.len() > x1a_rhs.len() {
                x1a_lhs
            } else {
                x1a_rhs
            };
            let x2a_new = if x2a_lhs.len() > x2a_rhs.len() {
                x2a_lhs
            } else {
                x2a_rhs
            };
            let ya_lhs = self.eval_grid(&x1a_new, &x2a_new);
            let ya_rhs = rhs.eval_grid(&x1a_new, &x2a_new);
            let ya_new = ya_lhs.iter().zip(ya_rhs.iter()).map(|(&a,&b)| a+b).collect::<Vec<_>>();
            (x1a_new,x2a_new,ya_new)
        };
        return Lagrange2dInterpolator::new(x1a_new,x2a_new,ya_new);
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> AddAssign<U> for Lagrange2dInterpolator<T,U> where i32: AsPrimitive<T> {
    fn add_assign(&mut self, rhs: U) {
        for y in &mut self.ya {
            for yy in y {
                *yy = *yy + rhs;
            }
        }
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> Sub<U> for Lagrange2dInterpolator<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange2dInterpolator<T,U>;
    fn sub(self, rhs: U) -> Self::Output {
        let (x1a,x2a,ya) = self.get_interp_data();
        let new_ya = ya.iter().flat_map(|y| y.iter()).map(|&e| e - rhs).collect::<Vec<_>>();
        return Lagrange2dInterpolator::new(x1a, x2a, new_ya);
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> Sub<Lagrange2dInterpolator<T,U>> for Lagrange2dInterpolator<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange2dInterpolator<T,U>;
    fn sub(self, rhs: Lagrange2dInterpolator<T,U>) -> Self::Output {
        let (x1a_lhs,x2a_lhs,ya_lhs) = self.get_interp_data();
        let (x1a_rhs,x2a_rhs,ya_rhs) = rhs.get_interp_data();
        
        let is_same_x1a = x1a_lhs.iter().zip(x1a_rhs.iter()).any(|(&a,&b)| (a-b).abs() > T::from(1e-12).unwrap()) && (x1a_lhs.len() == x1a_rhs.len());
        let is_same_x2a = x2a_lhs.iter().zip(x2a_rhs.iter()).any(|(&a,&b)| (a-b).abs() > T::from(1e-12).unwrap()) && (x2a_lhs.len() == x1a_rhs.len());

        // flatten y
        let ya_lhs = ya_lhs.iter().flat_map(|e| e.iter()).map(|&e| e).collect::<Vec<_>>();
        let ya_rhs = ya_rhs.iter().flat_map(|e| e.iter()).map(|&e| e).collect::<Vec<_>>();

        let (x1a_new,x2a_new,ya_new) = if is_same_x1a && is_same_x2a {
            (x1a_lhs,x2a_lhs,ya_lhs.iter().zip(ya_rhs.iter()).map(|(&a,&b)| a-b).collect::<Vec<_>>())
        } else {
            let x1a_new = if x1a_lhs.len() > x1a_rhs.len() {
                x1a_lhs
            } else {
                x1a_rhs
            };
            let x2a_new = if x2a_lhs.len() > x2a_rhs.len() {
                x2a_lhs
            } else {
                x2a_rhs
            };
            let ya_lhs = self.eval_grid(&x1a_new, &x2a_new);
            let ya_rhs = rhs.eval_grid(&x1a_new, &x2a_new);
            let ya_new = ya_lhs.iter().zip(ya_rhs.iter()).map(|(&a,&b)| a-b).collect::<Vec<_>>();
            (x1a_new,x2a_new,ya_new)
        };
        return Lagrange2dInterpolator::new(x1a_new,x2a_new,ya_new);
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> SubAssign<U> for Lagrange2dInterpolator<T,U> where i32: AsPrimitive<T> {
    fn sub_assign(&mut self, rhs: U) {
        for y in &mut self.ya {
            for yy in y {
                *yy = *yy - rhs;
            }
        }
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> Mul<U> for Lagrange2dInterpolator<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange2dInterpolator<T,U>;
    fn mul(self, rhs: U) -> Self::Output {
        let (x1a,x2a,ya) = self.get_interp_data();
        let new_ya = ya.iter().flat_map(|y| y.iter()).map(|&e| e*rhs).collect::<Vec<_>>();
        return Lagrange2dInterpolator::new(x1a, x2a, new_ya);
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> Mul<Lagrange2dInterpolator<T,U>> for Lagrange2dInterpolator<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange2dInterpolator<T,U>;
    fn mul(self, rhs: Lagrange2dInterpolator<T,U>) -> Self::Output {
        let (x1a_lhs,x2a_lhs,ya_lhs) = self.get_interp_data();
        let (x1a_rhs,x2a_rhs,ya_rhs) = rhs.get_interp_data();
        
        let is_same_x1a = x1a_lhs.iter().zip(x1a_rhs.iter()).any(|(&a,&b)| (a-b).abs() > T::from(1e-12).unwrap()) && (x1a_lhs.len() == x1a_rhs.len());
        let is_same_x2a = x2a_lhs.iter().zip(x2a_rhs.iter()).any(|(&a,&b)| (a-b).abs() > T::from(1e-12).unwrap()) && (x2a_lhs.len() == x1a_rhs.len());

        // flatten y
        let ya_lhs = ya_lhs.iter().flat_map(|e| e.iter()).map(|&e| e).collect::<Vec<_>>();
        let ya_rhs = ya_rhs.iter().flat_map(|e| e.iter()).map(|&e| e).collect::<Vec<_>>();

        let (x1a_new,x2a_new,ya_new) = if is_same_x1a && is_same_x2a {
            (x1a_lhs,x2a_lhs,ya_lhs.iter().zip(ya_rhs.iter()).map(|(&a,&b)| a*b).collect::<Vec<_>>())
        } else {
            let x1a_new = if x1a_lhs.len() > x1a_rhs.len() {
                x1a_lhs
            } else {
                x1a_rhs
            };
            let x2a_new = if x2a_lhs.len() > x2a_rhs.len() {
                x2a_lhs
            } else {
                x2a_rhs
            };
            let ya_lhs = self.eval_grid(&x1a_new, &x2a_new);
            let ya_rhs = rhs.eval_grid(&x1a_new, &x2a_new);
            let ya_new = ya_lhs.iter().zip(ya_rhs.iter()).map(|(&a,&b)| a*b).collect::<Vec<_>>();
            (x1a_new,x2a_new,ya_new)
        };
        return Lagrange2dInterpolator::new(x1a_new,x2a_new,ya_new);
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> MulAssign<U> for Lagrange2dInterpolator<T,U> where i32: AsPrimitive<T> {
    fn mul_assign(&mut self, rhs: U) {
        for y in &mut self.ya {
            for yy in y {
                *yy = *yy * rhs;
            }
        }
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> Div<U> for Lagrange2dInterpolator<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange2dInterpolator<T,U>;
    fn div(self, rhs: U) -> Self::Output {
        let (x1a,x2a,ya) = self.get_interp_data();
        let new_ya = ya.iter().flat_map(|y| y.iter()).map(|&e| e/rhs).collect::<Vec<_>>();
        return Lagrange2dInterpolator::new(x1a, x2a, new_ya);
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> Div<Lagrange2dInterpolator<T,U>> for Lagrange2dInterpolator<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange2dInterpolator<T,U>;
    fn div(self, rhs: Lagrange2dInterpolator<T,U>) -> Self::Output {
        let (x1a_lhs,x2a_lhs,ya_lhs) = self.get_interp_data();
        let (x1a_rhs,x2a_rhs,ya_rhs) = rhs.get_interp_data();
        
        let is_same_x1a = x1a_lhs.iter().zip(x1a_rhs.iter()).any(|(&a,&b)| (a-b).abs() > T::from(1e-12).unwrap()) && (x1a_lhs.len() == x1a_rhs.len());
        let is_same_x2a = x2a_lhs.iter().zip(x2a_rhs.iter()).any(|(&a,&b)| (a-b).abs() > T::from(1e-12).unwrap()) && (x2a_lhs.len() == x1a_rhs.len());

        // flatten y
        let ya_lhs = ya_lhs.iter().flat_map(|e| e.iter()).map(|&e| e).collect::<Vec<_>>();
        let ya_rhs = ya_rhs.iter().flat_map(|e| e.iter()).map(|&e| e).collect::<Vec<_>>();

        let (x1a_new,x2a_new,ya_new) = if is_same_x1a && is_same_x2a {
            (x1a_lhs,x2a_lhs,ya_lhs.iter().zip(ya_rhs.iter()).map(|(&a,&b)| a/b).collect::<Vec<_>>())
        } else {
            let x1a_new = if x1a_lhs.len() > x1a_rhs.len() {
                x1a_lhs
            } else {
                x1a_rhs
            };
            let x2a_new = if x2a_lhs.len() > x2a_rhs.len() {
                x2a_lhs
            } else {
                x2a_rhs
            };
            let ya_lhs = self.eval_grid(&x1a_new, &x2a_new);
            let ya_rhs = rhs.eval_grid(&x1a_new, &x2a_new);
            let ya_new = ya_lhs.iter().zip(ya_rhs.iter()).map(|(&a,&b)| a/b).collect::<Vec<_>>();
            (x1a_new,x2a_new,ya_new)
        };
        return Lagrange2dInterpolator::new(x1a_new,x2a_new,ya_new);
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> DivAssign<U> for Lagrange2dInterpolator<T,U> where i32: AsPrimitive<T> {
    fn div_assign(&mut self, rhs: U) {
        for y in &mut self.ya {
            for yy in y {
                *yy = *yy / rhs;
            }
        }
    }
}

// implementation of the basic operators for Lagrange1dInterpolatorVec
impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> Add<U> for Lagrange2dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange2dInterpolatorVec<T,U>;

    fn add(self, rhs: U) -> Self::Output {
        return Lagrange2dInterpolatorVec{
            lag2_interps: self.get_inner_interpolators().iter().map(|interp| interp.clone() + rhs).collect::<Vec<_>>()
        };
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> Add<Lagrange2dInterpolatorVec<T,U>> for Lagrange2dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange2dInterpolatorVec<T,U>;

    fn add(self, rhs: Lagrange2dInterpolatorVec<T,U>) -> Self::Output {
        let lhs = self.get_inner_interpolators();
        let rhs = rhs.get_inner_interpolators();
        return Lagrange2dInterpolatorVec{
            lag2_interps: lhs.iter().zip(rhs.iter()).map(|(i1,i2)| i1.clone()+i2.clone()).collect::<Vec<_>>()
        };
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> AddAssign<U> for Lagrange2dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    fn add_assign(&mut self, other: U) {
        for e in &mut self.lag2_interps {
            *e += other;
        }
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> Sub<U> for Lagrange2dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange2dInterpolatorVec<T,U>;

    fn sub(self, rhs: U) -> Self::Output {
        return Lagrange2dInterpolatorVec{
            lag2_interps: self.get_inner_interpolators().iter().map(|interp| interp.clone() - rhs).collect::<Vec<_>>()
        };
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> Sub<Lagrange2dInterpolatorVec<T,U>> for Lagrange2dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange2dInterpolatorVec<T,U>;

    fn sub(self, rhs: Lagrange2dInterpolatorVec<T,U>) -> Self::Output {
        let lhs = self.get_inner_interpolators();
        let rhs = rhs.get_inner_interpolators();
        return Lagrange2dInterpolatorVec{
            lag2_interps: lhs.iter().zip(rhs.iter()).map(|(i1,i2)| i1.clone()-i2.clone()).collect::<Vec<_>>()
        };
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> SubAssign<U> for Lagrange2dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    fn sub_assign(&mut self, other: U) {
        for e in &mut self.lag2_interps {
            *e -= other;
        }
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> Mul<U> for Lagrange2dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange2dInterpolatorVec<T,U>;

    fn mul(self, rhs: U) -> Self::Output {
        return Lagrange2dInterpolatorVec{
            lag2_interps: self.get_inner_interpolators().iter().map(|interp| interp.clone()*rhs).collect::<Vec<_>>()
        };
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> Mul<Lagrange2dInterpolatorVec<T,U>> for Lagrange2dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange2dInterpolatorVec<T,U>;

    fn mul(self, rhs: Lagrange2dInterpolatorVec<T,U>) -> Self::Output {
        let lhs = self.get_inner_interpolators();
        let rhs = rhs.get_inner_interpolators();
        return Lagrange2dInterpolatorVec{
            lag2_interps: lhs.iter().zip(rhs.iter()).map(|(i1,i2)| i1.clone()*i2.clone()).collect::<Vec<_>>()
        };
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> MulAssign<U> for Lagrange2dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    fn mul_assign(&mut self, other: U) {
        for e in &mut self.lag2_interps {
            *e *= other;
        }
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> Div<U> for Lagrange2dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange2dInterpolatorVec<T,U>;

    fn div(self, rhs: U) -> Self::Output {
        return Lagrange2dInterpolatorVec{
            lag2_interps: self.get_inner_interpolators().iter().map(|interp| interp.clone()/rhs).collect::<Vec<_>>()
        };
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> Div<Lagrange2dInterpolatorVec<T,U>> for Lagrange2dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    type Output = Lagrange2dInterpolatorVec<T,U>;

    fn div(self, rhs: Lagrange2dInterpolatorVec<T,U>) -> Self::Output {
        let lhs = self.get_inner_interpolators();
        let rhs = rhs.get_inner_interpolators();
        return Lagrange2dInterpolatorVec{
            lag2_interps: lhs.iter().zip(rhs.iter()).map(|(i1,i2)| i1.clone()/i2.clone()).collect::<Vec<_>>()
        };
    }
}

impl<T: LagRealTrait, U: LagComplexTrait + DivAssign<T> + MulAssign<T>> DivAssign<U> for Lagrange2dInterpolatorVec<T,U> where i32: AsPrimitive<T> {
    fn div_assign(&mut self, other: U) {
        for e in &mut self.lag2_interps {
            *e /= other;
        }
    }
}

// TESTS
#[cfg(test)]
pub mod lag2d_tests;