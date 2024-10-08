extern crate plotly;

use plotly::{color::NamedColor, common::{Marker, MarkerSymbol, Mode, Title, Line, DashType}, layout::{Axis,Layout}, Plot, Scatter,Surface};

pub fn lag1_compare_plot(xref: &Vec<f64>, yref: &Vec<f64>, xi: &Vec<f64>, yi: &Vec<f64>, xa: &Vec<f64>, ya: &Vec<f64>, title: String) {
    // 
    let layout = Layout::new().x_axis(Axis::new().title(Title::from("x"))).y_axis(Axis::new().title(Title::from("y"))).title(title);

    let rdata = Scatter::new(xref.clone(), yref.clone()).mode(Mode::Lines).name("Reference").line(Line::new().color(NamedColor::Green).width(6 as f64));
    let idata = Scatter::new(xi.clone(), yi.clone()).mode(Mode::Lines).name("Interpolated").line(Line::new().color(NamedColor::Red).width(3 as f64));
    let adata = Scatter::new(xa.clone(), ya.clone()).mode(Mode::Markers).marker(Marker::new().symbol(MarkerSymbol::Diamond).size(10).color(NamedColor::Blue)).name("Interpolator data");

    let mut plot = Plot::new();
    plot.add_trace(rdata);
    plot.add_trace(idata);
    plot.add_trace(adata);
    plot.set_layout(layout);
    plot.show();
}