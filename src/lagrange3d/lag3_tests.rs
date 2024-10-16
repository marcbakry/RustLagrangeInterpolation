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
}