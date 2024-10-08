use super::utilities::{partial_sum,LagRealTrait,LagComplexTrait};

pub fn lag1_eval<U,T>(xa: &Vec<T>, ya: &Vec<U>, x: T) -> U 
where 
T: LagRealTrait,
U: LagComplexTrait {
    // 
    let mut ns = xa.iter().map(|e| (*e-x).abs()).enumerate().min_by(|e1,e2| ((*e1).1).partial_cmp(&e2.1).unwrap()).map(|(idx,_)| idx).unwrap();
    let mut c = ya.clone();
    let mut d = ya.clone();
    let mut y = ya[ns];

    let n = xa.len();
    for m in 1..n {
        for i in 0..(n-m) {
            let ho = xa[i] - x;
            let hp = xa[i+m] - x;
            let w  = c[i+1] - d[i];
            d[i] = w*U::from(hp/(ho-hp)).unwrap();
            c[i] = w*U::from(ho/(ho-hp)).unwrap();
        }
        if 2*ns < n-m {
            y += c[ns];
        } else {
            y += d[ns-1];
            ns -= 1;
        }
    }
    // the end
    return y;
}

pub fn lag1_eval_vec<T,U>(xa: &Vec<T>, ya: &Vec<U>, x: &Vec<T>) -> Vec<U> 
where 
T: LagRealTrait,
U: LagComplexTrait {
    x.iter().map(|&e| lag1_eval(xa, ya, e)).collect::<Vec<U>>()
}

pub fn lag1_eval_derivative<T,U>(xa: &Vec<T>, ya: &Vec<U>, x: T) -> U 
where 
T: LagRealTrait,
U: LagComplexTrait {
    lag1_eval(
        xa,
        &ya.iter().enumerate().map(|(idx,&val)| val*U::from(partial_sum(xa,x,idx)).unwrap()).collect::<Vec<U>>(),
        x)
}

pub fn lag1_eval_derivative_vec<T,U>(xa: &Vec<T>, ya: &Vec<U>, x: &Vec<T>) -> Vec<U> 
where 
T: LagRealTrait,
U: LagComplexTrait {
    x.iter().map(|e| lag1_eval_derivative(xa, ya, *e)).collect::<Vec<U>>()
}