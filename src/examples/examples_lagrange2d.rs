//! This module provides some example functions illustrating the use of
//! Lagrange2dInterpolator and Lagrange1dInterpolator.
use crate::lagrange2d::Lagrange2dInterpolator;
use crate::utilities::gauss_chebyshev_nodes;
use crate::plot_utilities::*;
use std::time::Instant;

/// Example of interpolation of a function taken from the doc of the crate plotly.rs.
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

/// Comparison between the sequential and parallel versions of the evaluation of Lagrange2dInterpolator.
pub fn lag2_parallel_example() {
    println!("+-----------------------------------+");
    println!("| Running Lagrange 2d: SEQ. vs PAR. |");
    println!("+-----------------------------------+");
    let f = |x1: f64, x2: f64| 1.0/(x2*x2+5.0)*x2.sin() + 1.0/(x1*x1+5.0)*x1.cos();

    // create interpolator
    let (n1a,n2a) = (40,40);
    let x1a = gauss_chebyshev_nodes(&n1a, &-10.0, &10.0);
    let x2a = gauss_chebyshev_nodes(&n2a, &-10.0, &10.0);
    let mut ya: Vec<f64> = Vec::with_capacity(n1a*n2a);
    (0..n1a).for_each(|i1| (0..n2a).for_each(|i2| {
        ya.push(f(x1a[i1],x2a[i2]));
    }));
    let lag2_f = Lagrange2dInterpolator::new(x1a,x2a,ya);

    // 
    let (n1i,n2i) = (300,300);
    let (stp1i,stp2i) = (20.0/((n1i-1) as f64),20.0/((n1i-1) as f64));
    let x1i = (0..n1i).map(|i1| (i1 as f64)*stp1i - 10.0).collect::<Vec<f64>>();
    let x2i = (0..n2i).map(|i2| (i2 as f64)*stp2i - 10.0).collect::<Vec<f64>>();

    let x_arr = x1i.iter().flat_map(|x1| x2i.iter().map(move |x2| [*x1,*x2])).collect::<Vec<_>>();
    let x_tup = x1i.iter().flat_map(|x1| x2i.iter().map(move |x2| (*x1,*x2))).collect::<Vec<_>>();
    let (x1_vec,x2_vec): (Vec<_>,Vec<_>) = x_tup.iter().cloned().unzip();

    // gridded
    let start = Instant::now();
    let yi_seq = lag2_f.eval_grid(&x1i, &x2i);
    let end_seq = start.elapsed().as_millis();
    let start = Instant::now();
    let yi_par = lag2_f.par_eval_grid(&x1i, &x2i);
    let end_par = start.elapsed().as_millis();
    println!("Gridded interpolation");
    println!("- time seq.: {} (ms)",end_seq);
    println!("- time par.: {} (ms)",end_par);
    println!("- speed-up : {:.4}", (end_seq as f64)/(end_par as f64));
    println!("- error    : {}",yi_seq.iter().zip(yi_par.iter()).map(|(a,b)| (a-b).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap());

    // vec
    let start = Instant::now();
    let yi_seq = lag2_f.eval_vec(&x1_vec, &x2_vec);
    let end_seq = start.elapsed().as_millis();
    let start = Instant::now();
    let yi_par = lag2_f.par_eval_vec(&x1_vec, &x2_vec);
    let end_par = start.elapsed().as_millis();
    println!("Gridded interpolation");
    println!("- time seq.: {} (ms)",end_seq);
    println!("- time par.: {} (ms)",end_par);
    println!("- speed-up : {:.4}", (end_seq as f64)/(end_par as f64));
    println!("- error    : {}",yi_seq.iter().zip(yi_par.iter()).map(|(a,b)| (a-b).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap());

    // arr
    let start = Instant::now();
    let yi_seq = lag2_f.eval_arr(&x_arr);
    let end_seq = start.elapsed().as_millis();
    let start = Instant::now();
    let yi_par = lag2_f.par_eval_arr(&x_arr);
    let end_par = start.elapsed().as_millis();
    println!("Gridded interpolation");
    println!("- time seq.: {} (ms)",end_seq);
    println!("- time par.: {} (ms)",end_par);
    println!("- speed-up : {:.4}", (end_seq as f64)/(end_par as f64));
    println!("- error    : {}",yi_seq.iter().zip(yi_par.iter()).map(|(a,b)| (a-b).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap());
    
    // tup
    let start = Instant::now();
    let yi_seq = lag2_f.eval_tup(&x_tup);
    let end_seq = start.elapsed().as_millis();
    let start = Instant::now();
    let yi_par = lag2_f.par_eval_tup(&x_tup);
    let end_par = start.elapsed().as_millis();
    println!("Gridded interpolation");
    println!("- time seq.: {} (ms)",end_seq);
    println!("- time par.: {} (ms)",end_par);
    println!("- speed-up : {:.4}", (end_seq as f64)/(end_par as f64));
    println!("- error    : {}",yi_seq.iter().zip(yi_par.iter()).map(|(a,b)| (a-b).abs()).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap());
}