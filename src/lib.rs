//! This crate provides implementation for the gridded Lagrange interpolation of scalar- or 
//! vector-valued, real or complex, functions of one, two or three variables. Partial
//! derivatives of the underlying polynomials up to any order are available at nearly 
//! no-cost. Shorthands for the gradient and the jacobian matrix are also available 
//! when applicable. Many evaluation functions are provided depending on the kind of
//! the inputs. Moreover, the user may choose between the *sequential* or *parallel*
//! version of these functions.
//! 
//! The implementation relies on the standard `Vec` type for anything related to holding
//! the data. The parallelisation relies on [rayon.rs](https://crates.io/crates/rayon) 
//! and the plots are handled by [plotly.rs](https://crates.io/crates/plotly).
//! 
//! This crate is designed to be really easy to use. The high level interface features
//! a large amount of useful functions to evaluate or differentiate the underlying
//! Lagrange polynomials. It is a lot of sugar-coating interface around the low-level API
//! which can be found in the `lagX_utilies` modules.
//! 
//! 
//! # Arithmetic operators
//! 
//! This crate implements the four basic arithmetic operators (`+`,`-`,`*`,`/`)
//! for both scalars and other interpolators of the same kind. **See the doc of the corresponding traits
//! implementations for the `Lagrange1dInterpolator` structure for more details**, all other implementations 
//! work in the same fashion.
//! 
//! # How-To
//! 
//! The API is designed to be really easy to use. Most users will want to initialize a 
//! set of interpolation nodes and associate some value. Afterward, one can initialize 
//! an interpolator and it at some point in the domain. 
//! 
//! Some useful functions are provided in the `utilites` crate, for example to initialize 
//! a set of interpolation nodes.
//! 
//! While we prefer to refer to the `example` module, we give the general process for a 
//! simple one-dimensional interpolation of a scalar sine-function.
//! 
//! ```
//! // import the modules
//! use lagrange_interpolation::{lagrange1d::*,utilities::*};
//! use std::f64::consts::PI;
//! 
//! fn main() {
//!     // First, we define the function we wish to interpolate
//!     let f = |x: f64| f64::sin(2.0*PI*x.powi(2));
//!     // Then we create some interpolation data
//!     let n = 10;
//!     let (a,b) = (-0.5,0.5);
//!     let xa = gauss_chebyshev_nodes(&n,&a,&b); // avoids Gibbs effect
//!     let ya = xa.iter().map(|&x| f(a)).collect::<Vec<_>>();
//!     // We pack everything in a Lagrange univariate scalar interpolator.
//!     let i1d = Lagrange1dInterpolator(xa,ya);
//!     // Let's evaluate at some node (multi-node evaluation is also
//!     // available, see the corresponding doc).
//!     let x = 0.1;
//!     let val = i1d.eval(&x);
//!     // Let's compute the first derivative
//!     let i1d_dx = i1d.differentiate(); // less accurate interpolator of df/dx
//!     let val_dx = i1d_dx.eval(&x); // evaluate the interpolator of df/dx
//! }
//! ```
//! 
//! # Final disclaimer
//! 
//! This crate is developped by a Rust newbie on his spare time for learning purposes. 
//! Critics are welcome as long as they remain constructive.
//! 
//! As previously implied, this crate is not meant for performance, so anybody is encouraged to fork and improve the code.

pub mod utilities;
pub mod lagrange1d;
pub mod lagrange2d;
pub mod lagrange3d;

// misc
pub mod plot_utilities;
pub mod examples;