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

    pub fn eval(&self,x1: T, x2: T) -> U {
        lag2_eval(&self.x1a, &self.x2a, &self.ya, x1, x2)
    }

    pub fn eval_grid(&self, x1: &Vec<T>, x2: &Vec<T>) -> Vec<U> {
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

// TESTS
#[cfg(test)]
mod lagrange2d_tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    pub fn lag2_interpolation() {
        // 
        let f = |x1: f64,x2: f64| f64::cos(2.0*PI*x1.powi(2))*f64::sin(2.0*PI*x2).powi(2);
        let df_x1 = |x1: f64, x2: f64| -4.0*PI*x1*f64::sin(2.0*PI*x1.powi(2))*f64::sin(2.0*PI*x2).powi(2);
        let df_x2 = |x1: f64, x2: f64| f64::cos(2.0*PI*x1.powi(2))*4.0*PI*f64::sin(2.0*PI*x2)*f64::cos(2.0*PI*x2);
        // 
        let (n1a,n2a) = (21,31);
        let (a1,b1) = (0.0,1.0);
        let (a2,b2) = (0.0,1.0);
        let (stp1,stp2) = ((b1-a1)/((n1a-1) as f64),(b2-a2)/((n2a-1) as f64));
        let x1a = (0..n1a).map(|i| i as f64*stp1).collect::<Vec<_>>();
        let x2a = (0..n2a).map(|i| i as f64*stp2).collect::<Vec<_>>();
        let mut ya = Vec::with_capacity(x1a.len()*x2a.len());
        x1a.iter().for_each(|&x1| x2a.iter().for_each(|&x2| ya.push(f(x1,x2))));

        let lag2_f = Lagrange2dInterpolator::new(x1a,x2a,ya);
        let lag2_df_x1 = lag2_f.differentiate_x1();
        let lag2_df_x2 = lag2_f.differentiate_x2();

        //interpolated data
        let (n1i,n2i) = (50,51);
        let (stp1i,stp2i) = ((b1-a1)/((n1i-1) as f64),(b2-a2)/((n2i-1) as f64));
        let x1i = (0..n1i).map(|i| i as f64*stp1i).collect::<Vec<_>>();
        let x2i = (0..n2i).map(|i| i as f64*stp2i).collect::<Vec<_>>();

        let yi_f = lag2_f.eval_grid(&x1i, &x2i);
        let yi_df_x1 = lag2_df_x1.eval_grid(&x1i, &x2i);
        let yi_df_x2 = lag2_df_x2.eval_grid(&x1i, &x2i);

        // reference data
        let mut yref_f = Vec::with_capacity(n1i*n2i);
        x1i.iter().for_each(|&x1| x2i.iter().for_each(|&x2| yref_f.push(f(x1,x2))));

        let mut yref_df_x1 = Vec::with_capacity(n1i*n2i);
        x1i.iter().for_each(|&x1| x2i.iter().for_each(|&x2| yref_df_x1.push(df_x1(x1,x2))));
        
        let mut yref_df_x2 = Vec::with_capacity(n1i*n2i);
        x1i.iter().for_each(|&x1| x2i.iter().for_each(|&x2| yref_df_x2.push(df_x2(x1,x2))));

        // check accuracy with the maximum of the absolute error
        let err_f = yi_f.iter().zip(yref_f.iter()).map(|(ei,ef)| (ei-ef).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap();
        let err_df_x1 = yi_df_x1.iter().zip(yref_df_x1.iter()).map(|(ei,ef)| (ei-ef).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap();
        let err_df_x2 = yi_df_x2.iter().zip(yref_df_x2.iter()).map(|(ei,ef)| (ei-ef).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap();

        println!("Error 0-th order derivative         = {}",err_f);
        println!("Error 1-st order derivative (d/dx1) = {}",err_df_x1);
        println!("Error 1-st order derivative (d/dx2) = {}",err_df_x2);

        assert_eq!(lag2_f.order(),(n1a-1,n2a-1));
        assert_eq!(lag2_df_x1.order(),(n1a-2,n2a-1));
        assert_eq!(lag2_df_x2.order(),(n1a-1,n2a-2));
        assert_eq!(lag2_df_x1.diff_order(),(1,0));
        assert_eq!(lag2_df_x2.diff_order(),(0,1));
        assert!(err_f < 1e-6);
        assert!(err_df_x1 < 1e-3);
        assert!(err_df_x2 < 1e-3);
    }

    #[test]
    #[should_panic]
    pub fn lag2_input_size_mismatch() {
        let x1a = vec![1.0,2.0,3.0];
        let x2a = vec![1.0,2.0];
        let ya = vec![1.0; 5];

        let _lag2_f = Lagrange2dInterpolator::new(x1a,x2a,ya);
    }

    #[test]
    #[should_panic]
    pub fn lag2_null_size_x1a_input() {
        let x1a: Vec<f64> = Vec::new();
        let x2a = vec![1.0,2.0];
        let ya = vec![1.0,2.0];
        let _lag2_f = Lagrange2dInterpolator::new(x1a, x2a, ya);
    }
    
    #[test]
    #[should_panic]
    pub fn lag2_null_size_x2a_input() {
        let x1a: Vec<f64> = vec![1.0,2.0];
        let x2a = Vec::new();
        let ya = vec![1.0,2.0];
        let _lag2_f = Lagrange2dInterpolator::new(x1a, x2a, ya);
    }
    
    #[test]
    #[should_panic]
    pub fn lag2_null_size_ya_input() {
        let x1a: Vec<f64> = vec![1.0,2.0];
        let x2a = vec![1.0,2.0];
        let ya: Vec<f64> = Vec::new();
        let _lag2_f = Lagrange2dInterpolator::new(x1a, x2a, ya);
    }
    
    #[test]
    #[should_panic]
    pub fn lag2_duplicate_entries_x1a_input() {
        let x1a: Vec<f64> = vec![1.0,1.0,2.0];
        let x2a = vec![1.0,2.0];
        let ya: Vec<f64> = vec![1.0;6];
        let _lag2_f = Lagrange2dInterpolator::new(x1a, x2a, ya);
    }
    
    #[test]
    #[should_panic]
    pub fn lag2_duplicate_entries_x2a_input() {
        let x1a: Vec<f64> = vec![1.0,2.0];
        let x2a = vec![1.0,1.0,2.0];
        let ya: Vec<f64> = vec![1.0;6];
        let _lag2_f = Lagrange2dInterpolator::new(x1a, x2a, ya);
    }
}