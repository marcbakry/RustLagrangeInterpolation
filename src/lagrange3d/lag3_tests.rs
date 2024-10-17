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
    let (n1i,n2i,n3i) = (29,30,31);
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
    assert_eq!(lag3_f.order(),(n1a-1,n2a-1,n3a-1));
    assert_eq!(lag3_df_dx1.order(),(n1a-2,n2a-1,n3a-1));
    assert_eq!(lag3_df_dx2.order(),(n1a-1,n2a-2,n3a-1));
    assert_eq!(lag3_df_dx3.order(),(n1a-1,n2a-1,n3a-2));
    assert_eq!(lag3_df_dx1.diff_order(),(1,0,0));
    assert_eq!(lag3_df_dx2.diff_order(),(0,1,0));
    assert_eq!(lag3_df_dx3.diff_order(),(0,0,1));
}

#[test]
#[should_panic]
pub fn lag3_input_size_mismatch() {
    let x1a = vec![1.0,2.0,3.0];
    let x2a = vec![1.0,2.0];
    let x3a = vec![1.0,2.0,3.0];
    let ya = vec![1.0; 3];

    let _lag3_f = Lagrange3dInterpolator::new(x1a,x2a,x3a,ya);
}

#[test]
#[should_panic]
pub fn lag3_null_size_x1a_input() {
    let x1a = Vec::new();
    let x2a = vec![1.0,2.0,3.0];
    let x3a = vec![1.0,2.0,3.0];
    let ya = vec![1.0; 3];
    let _lag3_f = Lagrange3dInterpolator::new(x1a,x2a,x3a,ya);
}

#[test]
#[should_panic]
pub fn lag3_null_size_x2a_input() {
    let x1a = vec![1.0,2.0,3.0];
    let x2a = Vec::new();
    let x3a = vec![1.0,2.0,3.0];
    let ya = vec![1.0; 3];
    let _lag3_f = Lagrange3dInterpolator::new(x1a,x2a,x3a,ya);
}

#[test]
#[should_panic]
pub fn lag3_null_size_x3a_input() {
    let x1a = vec![1.0,2.0,3.0];
    let x2a = vec![1.0,2.0,3.0];
    let x3a = Vec::new();
    let ya = vec![1.0; 3];
    let _lag3_f = Lagrange3dInterpolator::new(x1a,x2a,x3a,ya);
}

#[test]
#[should_panic]
pub fn lag3_null_size_ya_input() {
    let x1a: Vec<f64> = vec![1.0,2.0];
    let x2a = vec![1.0,2.0];
    let x3a = vec![1.0,2.0];
    let ya: Vec<f64> = Vec::new();
    let _lag2_f = Lagrange3dInterpolator::new(x1a, x2a, x3a, ya);
}

#[test]
#[should_panic]
pub fn lag3_duplicate_entries_x1a_input() {
    let x1a: Vec<f64> = vec![1.0,1.0,2.0];
    let x2a = vec![1.0,2.0];
    let x3a = vec![1.0,2.0];
    let ya: Vec<f64> = vec![1.0;12];
    let _lag2_f = Lagrange3dInterpolator::new(x1a, x2a, x3a, ya);
}

#[test]
#[should_panic]
pub fn lag3_duplicate_entries_x2a_input() {
    let x1a: Vec<f64> = vec![1.0,2.0];
    let x2a = vec![1.0,1.0,2.0];
    let x3a = vec![1.0,2.0];
    let ya: Vec<f64> = vec![1.0;12];
    let _lag2_f = Lagrange3dInterpolator::new(x1a, x2a, x3a, ya);
}

#[test]
#[should_panic]
pub fn lag3_duplicate_entries_x3a_input() {
    let x1a: Vec<f64> = vec![1.0,2.0];
    let x2a = vec![1.0,2.0];
    let x3a = vec![1.0,1.0,2.0];
    let ya: Vec<f64> = vec![1.0;12];
    let _lag2_f = Lagrange3dInterpolator::new(x1a, x2a, x3a, ya);
}

#[test]
pub fn lag3_add_operators() {
    // 
    let f1 = |x1: f64, x2: f64, x3: f64| 1.0+x1+1.5*x2-3.0*(x1*x2+x1*x3-x1*x2);
    let f2 = |x1: f64, x2: f64, x3: f64| x1.powi(3)+x1.powi(2)*x2+x1*x2.powi(2)+x1.powi(2)*x3+x1*x3.powi(2)+x2.powi(3)+x2.powi(2)*x3+x2*x3.powi(2)+x3.powi(3);
    let f1plusf2 = |x1: f64, x2: f64, x3: f64| f1(x1,x2,x3) + f2(x1,x2,x3);
    // 
    let (a,b) = (-1.0,1.0);
    let (n1,n2) = (3,4);
    let (x1,x2) = (gauss_chebyshev_nodes(&n1, &a, &b), gauss_chebyshev_nodes(&n2, &a, &b));
    // 
    let ni = 20;
    let xi = linspace(&ni, &a, &b);
    // 
    let y1 = x1.iter().map(|x| x1.iter().map(|y| x1.iter().map(|z| f1(*x,*y,*z))).flatten()).flatten().collect::<Vec<_>>();
    let y2 = x2.iter().map(|x| x2.iter().map(|y| x2.iter().map(|z| f2(*x,*y,*z))).flatten()).flatten().collect::<Vec<_>>();
    let y12 = x2.iter().map(|x| x2.iter().map(|y| x2.iter().map(|z| f1plusf2(*x,*y,*z))).flatten()).flatten().collect::<Vec<_>>();
    // 
    let interp1 = Lagrange3dInterpolator::new(x1.clone(), x1.clone(), x1.clone(), y1);
    let interp2 = Lagrange3dInterpolator::new(x2.clone(), x2.clone(), x2.clone(), y2);
    let interp12 = interp1+interp2;
    let interp12_ref = Lagrange3dInterpolator::new(x2.clone(), x2.clone(), x2.clone(), y12);
    let (y_cal,y_ref) = (interp12.eval_grid(&xi, &xi, &xi),interp12_ref.eval_grid(&xi,&xi, &xi));
    let err = y_cal.iter().zip(y_ref.iter()).map(|(&a,&b)| (a-b).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap();
    assert!(err < 1e-14);
}

#[test]
pub fn lag3_sub_operators() {
    // 
    let f1 = |x1: f64, x2: f64, x3: f64| 1.0+x1+1.5*x2-3.0*(x1*x2+x1*x3-x1*x2);
    let f2 = |x1: f64, x2: f64, x3: f64| x1.powi(3)+x1.powi(2)*x2+x1*x2.powi(2)+x1.powi(2)*x3+x1*x3.powi(2)+x2.powi(3)+x2.powi(2)*x3+x2*x3.powi(2)+x3.powi(3);
    let f1plusf2 = |x1: f64, x2: f64, x3: f64| f1(x1,x2,x3) - f2(x1,x2,x3);
    // 
    let (a,b) = (-1.0,1.0);
    let (n1,n2) = (3,4);
    let (x1,x2) = (gauss_chebyshev_nodes(&n1, &a, &b), gauss_chebyshev_nodes(&n2, &a, &b));
    // 
    let ni = 20;
    let xi = linspace(&ni, &a, &b);
    // 
    let y1 = x1.iter().map(|x| x1.iter().map(|y| x1.iter().map(|z| f1(*x,*y,*z))).flatten()).flatten().collect::<Vec<_>>();
    let y2 = x2.iter().map(|x| x2.iter().map(|y| x2.iter().map(|z| f2(*x,*y,*z))).flatten()).flatten().collect::<Vec<_>>();
    let y12 = x2.iter().map(|x| x2.iter().map(|y| x2.iter().map(|z| f1plusf2(*x,*y,*z))).flatten()).flatten().collect::<Vec<_>>();
    // 
    let interp1 = Lagrange3dInterpolator::new(x1.clone(), x1.clone(), x1.clone(), y1);
    let interp2 = Lagrange3dInterpolator::new(x2.clone(), x2.clone(), x2.clone(), y2);
    let interp12 = interp1-interp2;
    let interp12_ref = Lagrange3dInterpolator::new(x2.clone(), x2.clone(), x2.clone(), y12);
    let (y_cal,y_ref) = (interp12.eval_grid(&xi, &xi, &xi),interp12_ref.eval_grid(&xi,&xi, &xi));
    let err = y_cal.iter().zip(y_ref.iter()).map(|(&a,&b)| (a-b).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap();
    assert!(err < 1e-14);
}

#[test]
pub fn lag3_mul_operators() {
    // 
    let f1 = |x1: f64, x2: f64, x3: f64| 1.0+x1+1.5*x2-3.0*(x1*x2+x1*x3-x1*x2);
    let f2 = |x1: f64, x2: f64, x3: f64| x1.powi(3)+x1.powi(2)*x2+x1*x2.powi(2)+x1.powi(2)*x3+x1*x3.powi(2)+x2.powi(3)+x2.powi(2)*x3+x2*x3.powi(2)+x3.powi(3);
    let f1plusf2 = |x1: f64, x2: f64, x3: f64| f1(x1,x2,x3) * f2(x1,x2,x3);
    // 
    let (a,b) = (-1.0,1.0);
    let (n1,n2) = (3,4);
    let (x1,x2) = (gauss_chebyshev_nodes(&n1, &a, &b), gauss_chebyshev_nodes(&n2, &a, &b));
    // 
    let ni = 20;
    let xi = linspace(&ni, &a, &b);
    // 
    let y1 = x1.iter().map(|x| x1.iter().map(|y| x1.iter().map(|z| f1(*x,*y,*z))).flatten()).flatten().collect::<Vec<_>>();
    let y2 = x2.iter().map(|x| x2.iter().map(|y| x2.iter().map(|z| f2(*x,*y,*z))).flatten()).flatten().collect::<Vec<_>>();
    let y12 = x2.iter().map(|x| x2.iter().map(|y| x2.iter().map(|z| f1plusf2(*x,*y,*z))).flatten()).flatten().collect::<Vec<_>>();
    // 
    let interp1 = Lagrange3dInterpolator::new(x1.clone(), x1.clone(), x1.clone(), y1);
    let interp2 = Lagrange3dInterpolator::new(x2.clone(), x2.clone(), x2.clone(), y2);
    let interp12 = interp1*interp2;
    let interp12_ref = Lagrange3dInterpolator::new(x2.clone(), x2.clone(), x2.clone(), y12);
    let (y_cal,y_ref) = (interp12.eval_grid(&xi, &xi, &xi),interp12_ref.eval_grid(&xi,&xi, &xi));
    let err = y_cal.iter().zip(y_ref.iter()).map(|(&a,&b)| (a-b).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap();
    assert!(err < 1e-14);
}

#[test]
pub fn lag3_div_operators() {
    // 
    let f1 = |x1: f64, x2: f64, x3: f64| 1.0+x1+1.5*x2-3.0*(x1*x2+x1*x3-x1*x2);
    let f2 = |x1: f64, x2: f64, x3: f64| 4.0+x1.powi(3)+x1.powi(2)*x2+x1*x2.powi(2)+x1.powi(2)*x3+x1*x3.powi(2)+x2.powi(3)+x2.powi(2)*x3+x2*x3.powi(2)+x3.powi(3);
    let f1plusf2 = |x1: f64, x2: f64, x3: f64| f1(x1,x2,x3) / f2(x1,x2,x3);
    // 
    let (a,b) = (-1.0,1.0);
    let (n1,n2) = (3,4);
    let (x1,x2) = (gauss_chebyshev_nodes(&n1, &a, &b), gauss_chebyshev_nodes(&n2, &a, &b));
    // 
    let ni = 20;
    let xi = linspace(&ni, &a, &b);
    // 
    let y1 = x1.iter().map(|x| x1.iter().map(|y| x1.iter().map(|z| f1(*x,*y,*z))).flatten()).flatten().collect::<Vec<_>>();
    let y2 = x2.iter().map(|x| x2.iter().map(|y| x2.iter().map(|z| f2(*x,*y,*z))).flatten()).flatten().collect::<Vec<_>>();
    let y12 = x2.iter().map(|x| x2.iter().map(|y| x2.iter().map(|z| f1plusf2(*x,*y,*z))).flatten()).flatten().collect::<Vec<_>>();
    // 
    let interp1 = Lagrange3dInterpolator::new(x1.clone(), x1.clone(), x1.clone(), y1);
    let interp2 = Lagrange3dInterpolator::new(x2.clone(), x2.clone(), x2.clone(), y2);
    let interp12 = interp1/interp2;
    let interp12_ref = Lagrange3dInterpolator::new(x2.clone(), x2.clone(), x2.clone(), y12);
    let (y_cal,y_ref) = (interp12.eval_grid(&xi, &xi, &xi),interp12_ref.eval_grid(&xi,&xi, &xi));
    let err = y_cal.iter().zip(y_ref.iter()).map(|(&a,&b)| (a-b).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap();
    assert!(err < 1e-14);
}