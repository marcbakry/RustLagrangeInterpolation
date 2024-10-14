use crate::lagrange1d::*;
use crate::lagrange2d::Lagrange2dInterpolator;
use crate::utilities::gauss_chebyshev_nodes;
use crate::plot_utilities::*;
use std::f64::consts::PI;
use std::time::Instant;

pub fn lag1_example_cosinus() {
    println!("+--------------------------------------+");
    println!("| Running Lagrange 1d: COSINE FUNCTION |");
    println!("+--------------------------------------+");
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
    println!("Done.");
}

pub fn lag1_example_quadratic_function() {
    println!("+-----------------------------------------+");
    println!("| Running Lagrange 1d: QUADRATIC FUNCTION |");
    println!("+-----------------------------------------+");
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

    println!("Done.");
}

pub fn lag1_parallel_example() {
    println!("+-----------------------------------+");
    println!("| Running Lagrange 1d: SEQ. vs PAR. |");
    println!("+-----------------------------------+");
    // function and first derivative
    let f = |x: f64| f64::cos(2.0*PI*x.powi(2));
    // interpolation data
    let (a,b) = (0.0,1.0);
    let na = 20;
    let xa_f = gauss_chebyshev_nodes(&na, &a, &b);
    let ya_f = xa_f.iter().map(|&x| f(x)).collect::<Vec<f64>>();
    let lag1_f = Lagrange1dInterpolator::new(xa_f.clone(),ya_f.clone());
    // interpolated data
    let ni = 10000000;
    let stpi = (b-a)/(ni-1) as f64;
    let xi = (0..ni).map(|i| i as f64*stpi).collect::<Vec<f64>>();
    // 
    let start = Instant::now();
    let yi_f_seq = lag1_f.eval_vec(&xi);
    let end_seq = start.elapsed().as_millis();
    let start = Instant::now();
    let yi_f_par = lag1_f.par_eval_vec(&xi);
    let end_par = start.elapsed().as_millis();
    println!("Time seq.: {} (ms)",end_seq);
    println!("Time par.: {} (ms)",end_par);
    println!("Speed-up : {}", (end_seq as  f64)/(end_par as f64));
    println!("error seq/par   : {}", yi_f_par.iter().zip(yi_f_seq.iter()).map(|(a,b)| (a-b).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap());
}

pub fn lag2_example() {
    println!("+------------------------------------+");
    println!("| Running Lagrange 2d: WAVY FUNCTION |");
    println!("+------------------------------------+");
    // 
    let f = |x1: f64, x2: f64| 1.0/(x2*x2+5.0)*x2.sin() + 1.0/(x1*x1+5.0)*x1.cos();

    // 
    let (n1a,n2a) = (40,40);
    let x1a = gauss_chebyshev_nodes(&n1a, &-10.0, &10.0);
    let x2a = gauss_chebyshev_nodes(&n2a, &-10.0, &10.0);
    let mut ya: Vec<f64> = Vec::with_capacity(n1a*n2a);
    (0..n1a).for_each(|i1| (0..n2a).for_each(|i2| {
        ya.push(f(x1a[i1],x2a[i2]));
    }));

    let lag2_f = Lagrange2dInterpolator::new(x1a.clone(),x2a.clone(),ya.clone());

    // 
    let (n1i,n2i) = (400,400);
    let(stp1i,stp2i) = (20.0/((n1i-1) as f64), 20.0/((n2i-1) as f64));
    let x1i = (0..n1i).map(|i1| (i1 as f64)*stp1i - 10.0).collect::<Vec<f64>>();
    let x2i = (0..n2i).map(|i2| (i2 as f64)*stp2i - 10.0).collect::<Vec<f64>>();

    let yi = lag2_f.eval_grid(&x1i, &x2i);

    // Reference
    let mut yref: Vec<f64> = Vec::with_capacity(n1i*n2i);
    (0..n1i).for_each(|i1| (0..n2i).for_each(|i2| {
        yref.push(f(x1i[i1],x2i[i2]));
    }));

    // 
    lag2_surface_plot(&x1a, &x2a, &ya, String::from("Interpolation data"));
    lag2_surface_plot(&x1i, &x2i, &yi, String::from("Interpolated data"));
    lag2_surface_plot(&x1i, &x2i, &yref, String::from("Reference data"));
    println!("Done.");
}