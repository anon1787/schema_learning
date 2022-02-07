use std::cmp::Ordering;

pub fn fp_cmp(x: f64, y: f64) -> Ordering {
    if x < y {
        Ordering::Less
    } else if x > y {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}
