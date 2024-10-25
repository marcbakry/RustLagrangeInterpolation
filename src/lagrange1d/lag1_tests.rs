
use super::*;
use std::f64::consts::PI;
use num::complex::*;

#[test]
fn lag1_real_interpolation() {
    // function and first derivative
    let f = |x: f64| f64::cos(2.0*PI*x.powi(2));
    let df = |x: f64| -4.0*PI*x*f64::sin(2.0*PI*x.powi(2));
    // interpolation data
    let (a,b) = (0.0,1.0);
    let na = 20;
    let xa = gauss_chebyshev_nodes(&na, &a, &b);
    let ya = xa.iter().map(|&x| f(x)).collect::<Vec<f64>>();
    let lag1_f = Lagrange1dInterpolator::new(xa,ya);
    let lag1_df = lag1_f.differentiate();

    // interpolated data
    let ni = 100;
    let stpi = (b-a)/(ni-1) as f64;
    let xi = (0..ni).map(|i| i as f64*stpi).collect::<Vec<f64>>();
    let yi_f = lag1_f.eval_vec(&xi);
    let yi_df = lag1_df.eval_vec(&xi);

    // reference data
    let yref_f = xi.iter().map(|&e| f(e)).collect::<Vec<f64>>();
    let yref_df = xi.iter().map(|&e| df(e)).collect::<Vec<f64>>();

    // check accuracy with the maximum of the absolute error
    let err_f = yi_f.iter().zip(yref_f.iter()).map(|(ei,ef)| (ei-ef).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap();
    let err_df = yi_df.iter().zip(yref_df.iter()).map(|(ei,ef)| (ei-ef).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap();

    println!("Error 0-th order derivative = {}",err_f);
    println!("Error 1-st order derivative = {}",err_df);

    assert_eq!(lag1_f.order(),na-1);
    assert_eq!(lag1_df.order(),na-2);
    assert_eq!(lag1_f.diff_order(),0);
    assert_eq!(lag1_df.diff_order(),1);
    assert!(err_f < 1e-6);
    assert!(err_df < 1e-3);
}

#[test]
fn lag1_complex_interpolation() {
    // function and first derivative
    let f = |x: f64| Complex::new(f64::cos(2.0*PI*x.powi(2)),f64::sin(2.0*PI*x.powi(2)));
    let df = |x: f64| Complex::new(-4.0*PI*x*f64::sin(2.0*PI*x.powi(2)), 4.0*PI*x*f64::cos(2.0*PI*x.powi(2)));
    // interpolation data
    let (a,b) = (0.0,1.0);
    let na = 20;
    let xa = gauss_chebyshev_nodes(&na, &a, &b);
    let ya = xa.iter().map(|&x| f(x)).collect::<Vec<_>>();
    let lag1_f = Lagrange1dInterpolator::new(xa,ya);
    let lag1_df = lag1_f.differentiate();

    // interpolated data
    let ni = 100;
    let stpi = (b-a)/(ni-1) as f64;
    let xi = (0..ni).map(|i| i as f64*stpi).collect::<Vec<f64>>();
    let yi_f = lag1_f.eval_vec(&xi);
    let yi_df = lag1_df.eval_vec(&xi);

    // reference data
    let yref_f = xi.iter().map(|&e| f(e)).collect::<Vec<_>>();
    let yref_df = xi.iter().map(|&e| df(e)).collect::<Vec<_>>();

    // check accuracy with the maximum of the absolute error
    let err_f = yi_f.iter().zip(yref_f.iter()).map(|(ei,ef)| (ei-ef).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap();
    let err_df = yi_df.iter().zip(yref_df.iter()).map(|(ei,ef)| (ei-ef).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap();

    println!("Error 0-th order derivative = {}",err_f);
    println!("Error 1-st order derivative = {}",err_df);

    assert_eq!(lag1_f.order(),na-1);
    assert_eq!(lag1_df.order(),na-2);
    assert_eq!(lag1_f.diff_order(),0);
    assert_eq!(lag1_df.diff_order(),1);
    assert!(err_f < 2.0*1e-6);
    assert!(err_df < 2.0*1e-3);
}

#[test]
#[should_panic]
pub fn lag1_input_size_mismatch() {
    let xa: Vec<f64> = vec![1.0,2.0,3.0];
    let mut ya = vec![1.0; 3];
    ya.pop();

    let _lag1_f = Lagrange1dInterpolator::new(xa,ya);
}

#[test]
#[should_panic]
pub fn lag1_null_size_xa_input() {
    let xa: Vec<f64> = Vec::new();
    let ya = vec![0.0; 3];
    let _lag1_f = Lagrange1dInterpolator::new(xa,ya);
}

#[test]
#[should_panic]
pub fn lag1_null_size_ya_input() {
    let xa: Vec<f64> = vec![1.0,2.0,3.0];
    let ya: Vec<f64> = Vec::new();
    let _lag1_f = Lagrange1dInterpolator::new(xa,ya);
}

#[test]
#[should_panic]
pub fn lag1_duplicate_entries() {
    let xa: Vec<f64> = vec![0.0; 3];
    let ya: Vec<f64> = xa.clone();
    let _lag1_f = Lagrange1dInterpolator::new(xa,ya);
}

#[test]
pub fn lag1_add_operators() {
    // 
    let f1 = |x: f64| f64::cos(2.0*PI*x);
    let f2 = |x: f64| x.powf(1.5);
    let f1plusf2 = |x: f64| f1(x) + f2(x);
    let (a,b) = (0.0,1.0);
    let (nm,np) = (9,10);
    let (xm,xp) = (gauss_chebyshev_nodes(&nm, &a, &b),gauss_chebyshev_nodes(&np, &a, &b));
    // 
    let ni = 100;
    let xi = linspace(&ni, &a, &b);

    // test add with same xa == xp
    let (y1,y2): (Vec<f64>,Vec<f64>) = xp.iter().map(|&x| (f1(x),f2(x))).unzip();
    let y12 = xp.iter().map(|&x| f1plusf2(x)).collect::<Vec<_>>();
    let interp1 = Lagrange1dInterpolator::new(xp.clone(), y1);
    let interp2 = Lagrange1dInterpolator::new(xp.clone(), y2);
    let interp12 = interp1+interp2;
    let interp12_ref = Lagrange1dInterpolator::new(xp.clone(),y12);
    let (y_cal,y_ref) = (interp12.eval_vec(&xi),interp12_ref.eval_vec(&xi));
    assert!(y_cal.iter().zip(y_ref.iter()).all(|(&a,&b)| (a-b).abs() < 1e-12));

    // test add with different xa
    let y1 = xm.iter().map(|&x| f1(x)).collect::<Vec<_>>();
    let y2 = xp.iter().map(|&x| f2(x)).collect::<Vec<_>>();
    let y12 = xp.iter().map(|&x| f1plusf2(x)).collect::<Vec<_>>();
    let interp1 = Lagrange1dInterpolator::new(xm.clone(), y1);
    let interp2 = Lagrange1dInterpolator::new(xp.clone(), y2);
    let interp12 = interp2+interp1;
    let interp12_ref = Lagrange1dInterpolator::new(xp.clone(),y12);
    let (y_cal,y_ref) = (interp12.eval_vec(&xi),interp12_ref.eval_vec(&xi));
    assert!(y_cal.iter().zip(y_ref.iter()).all(|(&a,&b)| (a-b).abs() < 1e-4));
}

#[test]
pub fn lag1_sub_operators() {
    // 
    let f1 = |x: f64| f64::cos(2.0*PI*x);
    let f2 = |x: f64| x.powf(1.5);
    let f1plusf2 = |x: f64| f1(x) - f2(x);
    let (a,b) = (0.0,1.0);
    let (nm,np) = (9,10);
    let (xm,xp) = (gauss_chebyshev_nodes(&nm, &a, &b),gauss_chebyshev_nodes(&np, &a, &b));
    // 
    let ni = 100;
    let xi = linspace(&ni, &a, &b);

    // test add with same xa == xp
    let (y1,y2): (Vec<f64>,Vec<f64>) = xp.iter().map(|&x| (f1(x),f2(x))).unzip();
    let y12 = xp.iter().map(|&x| f1plusf2(x)).collect::<Vec<_>>();
    let interp1 = Lagrange1dInterpolator::new(xp.clone(), y1);
    let interp2 = Lagrange1dInterpolator::new(xp.clone(), y2);
    let interp12 = interp1-interp2;
    let interp12_ref = Lagrange1dInterpolator::new(xp.clone(),y12);
    let (y_cal,y_ref) = (interp12.eval_vec(&xi),interp12_ref.eval_vec(&xi));
    assert!(y_cal.iter().zip(y_ref.iter()).all(|(&a,&b)| (a-b).abs() < 1e-12));

    // test add with different xa
    let y1 = xm.iter().map(|&x| f1(x)).collect::<Vec<_>>();
    let y2 = xp.iter().map(|&x| f2(x)).collect::<Vec<_>>();
    let y12 = xp.iter().map(|&x| f1plusf2(x)).collect::<Vec<_>>();
    let interp1 = Lagrange1dInterpolator::new(xm.clone(), y1);
    let interp2 = Lagrange1dInterpolator::new(xp.clone(), y2);
    let interp12 = interp1-interp2;
    let interp12_ref = Lagrange1dInterpolator::new(xp.clone(),y12);
    let (y_cal,y_ref) = (interp12.eval_vec(&xi),interp12_ref.eval_vec(&xi));
    assert!(y_cal.iter().zip(y_ref.iter()).all(|(&a,&b)| (a-b).abs() < 1e-4));
}

#[test]
pub fn lag1_mul_operators() {
    // 
    let f1 = |x: f64| f64::cos(2.0*PI*x);
    let f2 = |x: f64| x.powf(1.5);
    let f1plusf2 = |x: f64| f1(x)*f2(x);
    let (a,b) = (0.0,1.0);
    let (nm,np) = (9,10);
    let (xm,xp) = (gauss_chebyshev_nodes(&nm, &a, &b),gauss_chebyshev_nodes(&np, &a, &b));
    // 
    let ni = 100;
    let xi = linspace(&ni, &a, &b);

    // test add with same xa == xp
    let (y1,y2): (Vec<f64>,Vec<f64>) = xp.iter().map(|&x| (f1(x),f2(x))).unzip();
    let y12 = xp.iter().map(|&x| f1plusf2(x)).collect::<Vec<_>>();
    let interp1 = Lagrange1dInterpolator::new(xp.clone(), y1);
    let interp2 = Lagrange1dInterpolator::new(xp.clone(), y2);
    let interp12 = interp1*interp2;
    let interp12_ref = Lagrange1dInterpolator::new(xp.clone(),y12);
    let (y_cal,y_ref) = (interp12.eval_vec(&xi),interp12_ref.eval_vec(&xi));
    assert!(y_cal.iter().zip(y_ref.iter()).all(|(&a,&b)| (a-b).abs() < 1e-12));

    // test add with different xa
    let y1 = xm.iter().map(|&x| f1(x)).collect::<Vec<_>>();
    let y2 = xp.iter().map(|&x| f2(x)).collect::<Vec<_>>();
    let y12 = xp.iter().map(|&x| f1plusf2(x)).collect::<Vec<_>>();
    let interp1 = Lagrange1dInterpolator::new(xm.clone(), y1);
    let interp2 = Lagrange1dInterpolator::new(xp.clone(), y2);
    let interp12 = interp2*interp1;
    let interp12_ref = Lagrange1dInterpolator::new(xp.clone(),y12);
    let (y_cal,y_ref) = (interp12.eval_vec(&xi),interp12_ref.eval_vec(&xi));
    assert!(y_cal.iter().zip(y_ref.iter()).all(|(&a,&b)| (a-b).abs() < 1e-4));
}

#[test]
pub fn lag1_div_operators() {
    // 
    let f1 = |x: f64| f64::cos(2.0*PI*x);
    let f2 = |x: f64| 1.0+x.powf(1.5);
    let f1plusf2 = |x: f64| f1(x)/f2(x);
    let (a,b) = (0.0,1.0);
    let (nm,np) = (9,10);
    let (xm,xp) = (gauss_chebyshev_nodes(&nm, &a, &b),gauss_chebyshev_nodes(&np, &a, &b));
    // 
    let ni = 100;
    let xi = linspace(&ni, &a, &b);

    // test add with same xa == xp
    let (y1,y2): (Vec<f64>,Vec<f64>) = xp.iter().map(|&x| (f1(x),f2(x))).unzip();
    let y12 = xp.iter().map(|&x| f1plusf2(x)).collect::<Vec<_>>();
    let interp1 = Lagrange1dInterpolator::new(xp.clone(), y1);
    let interp2 = Lagrange1dInterpolator::new(xp.clone(), y2);
    let interp12 = interp1/interp2;
    let interp12_ref = Lagrange1dInterpolator::new(xp.clone(),y12);
    let (y_cal,y_ref) = (interp12.eval_vec(&xi),interp12_ref.eval_vec(&xi));
    assert!(y_cal.iter().zip(y_ref.iter()).all(|(&a,&b)| (a-b).abs() < 1e-12));

    // test add with different xa
    let y1 = xm.iter().map(|&x| f1(x)).collect::<Vec<_>>();
    let y2 = xp.iter().map(|&x| f2(x)).collect::<Vec<_>>();
    let y12 = xp.iter().map(|&x| f1plusf2(x)).collect::<Vec<_>>();
    let interp1 = Lagrange1dInterpolator::new(xm.clone(), y1);
    let interp2 = Lagrange1dInterpolator::new(xp.clone(), y2);
    let interp12 = interp1/interp2;
    let interp12_ref = Lagrange1dInterpolator::new(xp.clone(),y12);
    let (y_cal,y_ref) = (interp12.eval_vec(&xi),interp12_ref.eval_vec(&xi));
    assert!(y_cal.iter().zip(y_ref.iter()).all(|(&a,&b)| (a-b).abs() < 1e-4));
}

#[test]
pub fn lag1_basis_as_interpolators() {
    let (a,b) = (0.0,1.0);
    let na = 20;
    let xa = gauss_chebyshev_nodes(&na, &a, &b);
    let ya = xa.iter().map(|&x| x).collect::<Vec<f64>>();
    let i1d = Lagrange1dInterpolator::new(xa.clone(),ya);
    let i1d_basis = i1d.lagrange_basis();
    for i in 0..i1d_basis.len() {
        for j in 0..xa.len() {
            if i != j {
                assert!(i1d_basis[i].eval(&xa[j]).abs() < 1e-14);
            } else {
                assert!((i1d_basis[i].eval(&xa[j]) - 1.0).abs() < 1e-14);
            }
        }
    }
}

#[test]
pub fn lag1_lagrange_basis_and_derivatives() {
    // 
    let xa = vec![0.0,1.0,2.0];
    let ya = vec![2.0;3];
    let i1d = Lagrange1dInterpolator::new(xa.clone(), ya);
    // references
    let f0 = |x: f64| (x-xa[1])*(x-xa[2])/((xa[0]-xa[1])*(xa[0]-xa[2]));
    let f1 = |x: f64| (x-xa[0])*(x-xa[2])/((xa[1]-xa[0])*(xa[1]-xa[2]));
    let f2 = |x: f64| (x-xa[0])*(x-xa[1])/((xa[2]-xa[0])*(xa[2]-xa[1]));

    let df0_dx = |x: f64| (2.0*x - xa[1] - xa[2])/((xa[0]-xa[1])*(xa[0]-xa[2]));
    let df1_dx = |x: f64| (2.0*x - xa[0] - xa[2])/((xa[1]-xa[0])*(xa[1]-xa[2]));
    let df2_dx = |x: f64| (2.0*x - xa[0] - xa[1])/((xa[2]-xa[0])*(xa[2]-xa[1]));
    // 
    let ni = 101;
    let xi = linspace(&ni, &xa[0], &xa[2]);
    // ---------------------------------
    // evaluation of the basis functions
    // ---------------------------------
    let basis_value = i1d.eval_basis_vec(&xi);
    // 1st basis function
    let val_cal = &basis_value[0];
    let val_ref = xi.iter().map(|&x| f0(x)).collect::<Vec<_>>();
    assert!(val_cal.iter().zip(val_ref.iter()).map(|(&a,&b)| (a-b).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap() < 1e-14);
    // 2nd basis function
    let val_cal = &basis_value[1];
    let val_ref = xi.iter().map(|&x| f1(x)).collect::<Vec<_>>();
    assert!(val_cal.iter().zip(val_ref.iter()).map(|(&a,&b)| (a-b).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap() < 1e-14);
    // 3rd basis function
    let val_cal = &basis_value[2];
    let val_ref = xi.iter().map(|&x| f2(x)).collect::<Vec<_>>();
    assert!(val_cal.iter().zip(val_ref.iter()).map(|(&a,&b)| (a-b).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap() < 1e-14);

    // -----------------------------
    // evaluation of the derivatives
    // -----------------------------
    let i1d_basis_diff = i1d.differentiate_lagrange_basis();
    // 1st basis function
    let val_cal = i1d_basis_diff[0].eval_vec(&xi);
    let val_ref = xi.iter().map(|&x| df0_dx(x)).collect::<Vec<_>>();
    assert!(val_cal.iter().zip(val_ref.iter()).map(|(&a,&b)| (a-b).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap() < 1e-14);
    // 2nd basis function
    let val_cal = i1d_basis_diff[1].eval_vec(&xi);
    let val_ref = xi.iter().map(|&x| df1_dx(x)).collect::<Vec<_>>();
    assert!(val_cal.iter().zip(val_ref.iter()).map(|(&a,&b)| (a-b).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap() < 1e-14);
    // 3rd basis function
    let val_cal = i1d_basis_diff[2].eval_vec(&xi);
    let val_ref = xi.iter().map(|&x| df2_dx(x)).collect::<Vec<_>>();
    assert!(val_cal.iter().zip(val_ref.iter()).map(|(&a,&b)| (a-b).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap() < 1e-14);
}