use crate::lagrange1d::*;
use crate::utilities::gauss_chebyshev_nodes;
use crate::plot_utilities::*;
use std::f64::consts::PI;

pub fn lag1_example_cosinus() {
    // function and first derivative
    let f = |x: f64| f64::cos(2.0*PI*x.powi(2));
    let df = |x: f64| -4.0*PI*x*f64::sin(2.0*PI*x.powi(2));
    // interpolation data
    let (a,b) = (0.0,1.0);
    let na = 20;
    let xa_f = gauss_chebyshev_nodes(&na, &a, &b);
    let ya_f = xa_f.iter().map(|&x| f(x)).collect::<Vec<f64>>();
    let lag1_f = Lagrange1dInterpolator::new(xa_f.clone(),ya_f.clone());
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

    lag1_compare_plot(&xi, &yref_f, &xi, &yi_f, &xa_f, &ya_f, String::from("Comparison f(x) = cos(2 pi x^2)"));

    let (xa_df,ya_df) = lag1_df.get_interp_data();
    lag1_compare_plot(&xi, &yref_df, &xi, &yi_df, &xa_df, &ya_df, String::from("Comparison df(x)/dx = -2 pi x sin(2 pi x^2)"));
}

pub fn lag1_example_quadratic_function() {
    let f = |x: f64| x.powi(2);
    let df = |x: f64| 2.0*x;
    let ddf = |_: f64| 2.0;
    let dddf = |_: f64| 0.0;

    let (a,b) = (0.0,1.0);
    let na = 3; // second order polynom
    let xa_f = gauss_chebyshev_nodes(&na, &a, &b);
    let ya_f = xa_f.iter().map(|&x| f(x)).collect::<Vec<f64>>();
    let lag1_f = Lagrange1dInterpolator::new(xa_f.clone(),ya_f.clone());
    let lag1_df = lag1_f.differentiate();
    let lag1_ddf = lag1_df.differentiate();
    let lag1_dddf = lag1_ddf.differentiate();

    // interpolated data
    let ni = 20;
    let stpi = (b-a)/(ni-1) as f64;
    let xi = (0..ni).map(|i| i as f64*stpi).collect::<Vec<f64>>();
    let yi_f = lag1_f.eval_vec(&xi);
    let yi_df = lag1_df.eval_vec(&xi);
    let yi_ddf = lag1_ddf.eval_vec(&xi);
    let yi_dddf = lag1_dddf.eval_vec(&xi);

    // reference data
    let yref_f = xi.iter().map(|&e| f(e)).collect::<Vec<f64>>();
    let yref_df = xi.iter().map(|&e| df(e)).collect::<Vec<f64>>();
    let yref_ddf = xi.iter().map(|&e| ddf(e)).collect::<Vec<f64>>();
    let yref_dddf = xi.iter().map(|&e| dddf(e)).collect::<Vec<f64>>();

    lag1_compare_plot(&xi, &yref_f, &xi, &yi_f, &xa_f, &ya_f, String::from("Comparison f(x) = x^2"));

    let (xa_df,ya_df) = lag1_df.get_interp_data();
    lag1_compare_plot(&xi, &yref_df, &xi, &yi_df, &xa_df, &ya_df, String::from("Comparison df(x)/dx = 2.0*x"));
    
    let (xa_ddf,ya_ddf) = lag1_ddf.get_interp_data();
    lag1_compare_plot(&xi, &yref_ddf, &xi, &yi_ddf, &xa_ddf, &ya_ddf, String::from("Comparison d^2f(x)/dx^2 = 2.0"));
    
    let (xa_dddf,ya_dddf) = lag1_dddf.get_interp_data();
    lag1_compare_plot(&xi, &yref_dddf, &xi, &yi_dddf, &xa_dddf, &ya_dddf, String::from("Comparison d^3f(x)/dx^3 = 0.0"));
}