//! Interface module for running the examples.
pub mod examples_lagrange1d;
pub mod examples_lagrange2d;

use examples_lagrange1d::*;
use examples_lagrange2d::*;

pub fn run_lag1_examples() {
    lag1_example_cosinus();
    lag1_example_quadratic_function();
    lag1_parallel_example();
}

pub fn run_lag2_examples() {
    lag2_example();
    lag2_parallel_example();
}

pub fn run_all_examples() {
    run_lag1_examples();
    run_lag2_examples();
}