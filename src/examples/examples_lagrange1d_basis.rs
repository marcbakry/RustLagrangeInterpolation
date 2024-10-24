use crate::{lagrange1d::*, utilities::gauss_chebyshev_nodes};
use crate::utilities::linspace;
use plotly::{color::NamedColor, common::{Line, Marker, MarkerSymbol, Mode, Title}, layout::{Axis,Layout}, Plot, Scatter};

pub fn lag1_plot_order3_basis() {
    // 
    let (a,b) = (0.0,1.0);
    let n = 4;
    let xa = gauss_chebyshev_nodes(&n,&a,&b);
    let ya = vec![1.0;n];
    let i1d = Lagrange1dInterpolator::new(xa.clone(),ya);

    let ni = 1001;
    let xi = linspace(&ni, &a, &b);
    let basis_values = i1d.eval_basis_vec(&xi);

    let labels = (0..n).map(|i| format!("Basis n° {i}")).collect::<Vec<_>>();

    let mut lines = Vec::with_capacity(n);
    for i in 0..n {
        let data = Scatter::new(xi.clone(), basis_values[i].clone()).mode(Mode::Lines).name(labels[i].clone()).line(Line::new().width(2 as f64));
        lines.push(data);
    }

    let layout = Layout::new().x_axis(Axis::new().title(Title::from("x"))).y_axis(Axis::new().title(Title::from("y"))).title("3rd order Lagrange basis functions with Gauss-Chebyshev nodes");

    let mut plot = Plot::new();

    for i in 0..n {
        plot.add_trace(lines[i].clone());
    }
    plot.add_trace(
        Scatter::new(xa.clone(),vec![0.0;n]).mode(Mode::Markers).marker(Marker::new().symbol(MarkerSymbol::Diamond).size(10).color(NamedColor::Indigo)).show_legend(false)
    );
    plot.add_trace(
        Scatter::new(xa.clone(),vec![1.0;n]).mode(Mode::Markers).marker(Marker::new().symbol(MarkerSymbol::Diamond).size(10).color(NamedColor::Indigo)).show_legend(false)
    );

    plot.set_layout(layout);
    plot.show();
}