
pub struct Digit {
  pub class: u8,
  pub grid: Vec<f64>
}

impl Digit {
  pub fn new(class: u8, grid: Vec<f64>) -> Digit {
    if class > 9 || grid.len() != 64 {
      panic!("Incorrect input values.");
    }

    Digit { class, grid }
  }
}
