
pub struct Digit {
  pub class: f64,
  pub grid: Vec<f64>
}

impl Digit {
  pub fn new(class: f64, grid: Vec<f64>) -> Digit {
    if class > 9_f64 || class < 0_f64 || grid.len() != 64 {
      panic!("Incorrect input values.");
    }

    Digit { class, grid }
  }
}
