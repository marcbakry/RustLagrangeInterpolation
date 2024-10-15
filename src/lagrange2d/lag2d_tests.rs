
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

#[test]
pub fn lag2_add_operators() {
    let f1 = |x: f64, y: f64| f64::cos(2.0*PI*x)*f64::sin(PI*y);
    let f2 = |x: f64, y: f64| 1.0 + x + 1.5*y - 3.0*x*y + x.powi(2)+f64::sqrt(2.0)*y.powi(2);
    let f1plusf2 = |x: f64, y: f64| f1(x,y) + f2(x,y);
    let (a,b) = (0.0,1.0);
    let (n1,n2) = (10,3);
    let x1 = gauss_chebyshev_nodes(&n1, &a, &b); 
    let x2 = gauss_chebyshev_nodes(&n2, &a, &b);
    // 
    let ni = 50;
    let xi = linspace(&ni, &a, &b);

    // test add with matching x1a/x2a
    let y1 = x1.iter().map(|x| x1.iter().map(|y| f1(*x,*y))).flatten().collect::<Vec<_>>();
    let y2 = x1.iter().map(|x| x1.iter().map(|y| f2(*x,*y))).flatten().collect::<Vec<_>>();
    let y12 = x1.iter().map(|x| x1.iter().map(|y| f1plusf2(*x,*y))).flatten().collect::<Vec<_>>();

    let interp1 = Lagrange2dInterpolator::new(x1.clone(),x1.clone(),y1);
    let interp2 = Lagrange2dInterpolator::new(x1.clone(),x1.clone(),y2);
    let interp12 = interp1 + interp2;
    let interp12_ref = Lagrange2dInterpolator::new(x1.clone(),x1.clone(),y12);
    let (y_cal,y_ref) = (interp12.eval_grid(&xi, &xi),interp12_ref.eval_grid(&xi,&xi));
    let err = y_cal.iter().zip(y_ref.iter()).map(|(&a,&b)| (a-b).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap();
    assert!(err < 1e-14);

    // test add with non-matching x1a or x2a
    let y1 = x1.iter().map(|x| x1.iter().map(|y| f1(*x,*y))).flatten().collect::<Vec<_>>();
    let y2 = x2.iter().map(|x| x2.iter().map(|y| f2(*x,*y))).flatten().collect::<Vec<_>>();
    let y12 = x1.iter().map(|x| x1.iter().map(|y| f1plusf2(*x,*y))).flatten().collect::<Vec<_>>();

    let interp1 = Lagrange2dInterpolator::new(x1.clone(),x1.clone(),y1);
    let interp2 = Lagrange2dInterpolator::new(x2.clone(),x2.clone(),y2);
    let interp12 = interp1 + interp2;
    let interp12_ref = Lagrange2dInterpolator::new(x1.clone(),x1.clone(),y12);
    let (y_cal,y_ref) = (interp12.eval_grid(&xi, &xi),interp12_ref.eval_grid(&xi,&xi));
    let err = y_cal.iter().zip(y_ref.iter()).map(|(&a,&b)| (a-b).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap();
    assert!(err < 1e-14);
}

#[test]
pub fn lag2_sub_operators() {
    let f1 = |x: f64, y: f64| f64::cos(2.0*PI*x)*f64::sin(PI*y);
    let f2 = |x: f64, y: f64| 1.0 + x + 1.5*y - 3.0*x*y + x.powi(2)+f64::sqrt(2.0)*y.powi(2);
    let f1plusf2 = |x: f64, y: f64| f1(x,y) - f2(x,y);
    let (a,b) = (0.0,1.0);
    let (n1,n2) = (10,3);
    let x1 = gauss_chebyshev_nodes(&n1, &a, &b); 
    let x2 = gauss_chebyshev_nodes(&n2, &a, &b);
    // 
    let ni = 50;
    let xi = linspace(&ni, &a, &b);

    // test add with matching x1a/x2a
    let y1 = x1.iter().map(|x| x1.iter().map(|y| f1(*x,*y))).flatten().collect::<Vec<_>>();
    let y2 = x1.iter().map(|x| x1.iter().map(|y| f2(*x,*y))).flatten().collect::<Vec<_>>();
    let y12 = x1.iter().map(|x| x1.iter().map(|y| f1plusf2(*x,*y))).flatten().collect::<Vec<_>>();

    let interp1 = Lagrange2dInterpolator::new(x1.clone(),x1.clone(),y1);
    let interp2 = Lagrange2dInterpolator::new(x1.clone(),x1.clone(),y2);
    let interp12 = interp1 - interp2;
    let interp12_ref = Lagrange2dInterpolator::new(x1.clone(),x1.clone(),y12);
    let (y_cal,y_ref) = (interp12.eval_grid(&xi, &xi),interp12_ref.eval_grid(&xi,&xi));
    let err = y_cal.iter().zip(y_ref.iter()).map(|(&a,&b)| (a-b).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap();
    assert!(err < 1e-14);

    // test add with non-matching x1a or x2a
    let y1 = x1.iter().map(|x| x1.iter().map(|y| f1(*x,*y))).flatten().collect::<Vec<_>>();
    let y2 = x2.iter().map(|x| x2.iter().map(|y| f2(*x,*y))).flatten().collect::<Vec<_>>();
    let y12 = x1.iter().map(|x| x1.iter().map(|y| f1plusf2(*x,*y))).flatten().collect::<Vec<_>>();

    let interp1 = Lagrange2dInterpolator::new(x1.clone(),x1.clone(),y1);
    let interp2 = Lagrange2dInterpolator::new(x2.clone(),x2.clone(),y2);
    let interp12 = interp1 - interp2;
    let interp12_ref = Lagrange2dInterpolator::new(x1.clone(),x1.clone(),y12);
    let (y_cal,y_ref) = (interp12.eval_grid(&xi, &xi),interp12_ref.eval_grid(&xi,&xi));
    let err = y_cal.iter().zip(y_ref.iter()).map(|(&a,&b)| (a-b).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap();
    assert!(err < 1e-14);
}

#[test]
pub fn lag2_mul_operators() {
    let f1 = |x: f64, y: f64| f64::cos(2.0*PI*x)*f64::sin(PI*y);
    let f2 = |x: f64, y: f64| 1.0 + x + 1.5*y - 3.0*x*y + x.powi(2)+f64::sqrt(2.0)*y.powi(2);
    let f1plusf2 = |x: f64, y: f64| f1(x,y) * f2(x,y);
    let (a,b) = (0.0,1.0);
    let (n1,n2) = (10,3);
    let x1 = gauss_chebyshev_nodes(&n1, &a, &b); 
    let x2 = gauss_chebyshev_nodes(&n2, &a, &b);
    // 
    let ni = 50;
    let xi = linspace(&ni, &a, &b);

    // test add with matching x1a/x2a
    let y1 = x1.iter().map(|x| x1.iter().map(|y| f1(*x,*y))).flatten().collect::<Vec<_>>();
    let y2 = x1.iter().map(|x| x1.iter().map(|y| f2(*x,*y))).flatten().collect::<Vec<_>>();
    let y12 = x1.iter().map(|x| x1.iter().map(|y| f1plusf2(*x,*y))).flatten().collect::<Vec<_>>();

    let interp1 = Lagrange2dInterpolator::new(x1.clone(),x1.clone(),y1);
    let interp2 = Lagrange2dInterpolator::new(x1.clone(),x1.clone(),y2);
    let interp12 = interp1 * interp2;
    let interp12_ref = Lagrange2dInterpolator::new(x1.clone(),x1.clone(),y12);
    let (y_cal,y_ref) = (interp12.eval_grid(&xi, &xi),interp12_ref.eval_grid(&xi,&xi));
    let err = y_cal.iter().zip(y_ref.iter()).map(|(&a,&b)| (a-b).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap();
    assert!(err < 1e-14);

    // test add with non-matching x1a or x2a
    let y1 = x1.iter().map(|x| x1.iter().map(|y| f1(*x,*y))).flatten().collect::<Vec<_>>();
    let y2 = x2.iter().map(|x| x2.iter().map(|y| f2(*x,*y))).flatten().collect::<Vec<_>>();
    let y12 = x1.iter().map(|x| x1.iter().map(|y| f1plusf2(*x,*y))).flatten().collect::<Vec<_>>();

    let interp1 = Lagrange2dInterpolator::new(x1.clone(),x1.clone(),y1);
    let interp2 = Lagrange2dInterpolator::new(x2.clone(),x2.clone(),y2);
    let interp12 = interp1 * interp2;
    let interp12_ref = Lagrange2dInterpolator::new(x1.clone(),x1.clone(),y12);
    let (y_cal,y_ref) = (interp12.eval_grid(&xi, &xi),interp12_ref.eval_grid(&xi,&xi));
    let err = y_cal.iter().zip(y_ref.iter()).map(|(&a,&b)| (a-b).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap();
    assert!(err < 1e-14);
}

#[test]
pub fn lag2_div_operators() {
    let f1 = |x: f64, y: f64| f64::cos(2.0*PI*x)*f64::sin(PI*y);
    let f2 = |x: f64, y: f64| 1.0 + x + 1.5*y - 3.0*x*y + x.powi(2)+f64::sqrt(2.0)*y.powi(2);
    let f1plusf2 = |x: f64, y: f64| f1(x,y) / f2(x,y);
    let (a,b) = (0.0,1.0);
    let (n1,n2) = (10,3);
    let x1 = gauss_chebyshev_nodes(&n1, &a, &b); 
    let x2 = gauss_chebyshev_nodes(&n2, &a, &b);
    // 
    let ni = 50;
    let xi = linspace(&ni, &a, &b);

    // test add with matching x1a/x2a
    let y1 = x1.iter().map(|x| x1.iter().map(|y| f1(*x,*y))).flatten().collect::<Vec<_>>();
    let y2 = x1.iter().map(|x| x1.iter().map(|y| f2(*x,*y))).flatten().collect::<Vec<_>>();
    let y12 = x1.iter().map(|x| x1.iter().map(|y| f1plusf2(*x,*y))).flatten().collect::<Vec<_>>();

    let interp1 = Lagrange2dInterpolator::new(x1.clone(),x1.clone(),y1);
    let interp2 = Lagrange2dInterpolator::new(x1.clone(),x1.clone(),y2);
    let interp12 = interp1 / interp2;
    let interp12_ref = Lagrange2dInterpolator::new(x1.clone(),x1.clone(),y12);
    let (y_cal,y_ref) = (interp12.eval_grid(&xi, &xi),interp12_ref.eval_grid(&xi,&xi));
    let err = y_cal.iter().zip(y_ref.iter()).map(|(&a,&b)| (a-b).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap();
    assert!(err < 1e-14);

    // test add with non-matching x1a or x2a
    let y1 = x1.iter().map(|x| x1.iter().map(|y| f1(*x,*y))).flatten().collect::<Vec<_>>();
    let y2 = x2.iter().map(|x| x2.iter().map(|y| f2(*x,*y))).flatten().collect::<Vec<_>>();
    let y12 = x1.iter().map(|x| x1.iter().map(|y| f1plusf2(*x,*y))).flatten().collect::<Vec<_>>();

    let interp1 = Lagrange2dInterpolator::new(x1.clone(),x1.clone(),y1);
    let interp2 = Lagrange2dInterpolator::new(x2.clone(),x2.clone(),y2);
    let interp12 = interp1 / interp2;
    let interp12_ref = Lagrange2dInterpolator::new(x1.clone(),x1.clone(),y12);
    let (y_cal,y_ref) = (interp12.eval_grid(&xi, &xi),interp12_ref.eval_grid(&xi,&xi));
    let err = y_cal.iter().zip(y_ref.iter()).map(|(&a,&b)| (a-b).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap();
    assert!(err < 1e-14);
}