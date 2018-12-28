extern crate rand;

mod reader;
mod neural_network;

use reader::digit::Digit;
use neural_network::Network;

fn main() {
  let digits: Vec<Digit> = reader::read_digits();
  let training_set: Vec<(u8, Vec<f64>)> = digits.iter()
    .map(|digit| (digit.class, digit.grid.clone()))
    .collect();

  let mut network = Network::new(vec!(64, 12, 12, 10));

  println!("Network classify {}", network.train(training_set));
}

#[cfg(test)]
mod tests {
  use super::neural_network::Network;

  #[test]
  fn xor_gate() {
    let network: Network = Network::from(vec!(
      vec!(
        (-0.1_f64, vec!(0.1_f64, 0.1_f64)), (0_f64, vec!(0.9_f64, 0.9_f64))
      ),
      vec!(
        (-0.01_f64, vec!(-10_f64, 1_f64)), (0.01_f64, vec!(100_f64, -1_f64))
      )
    ));

    assert!(network.classify(vec!(0_f64, 1_f64)) == 0);
    assert!(network.classify(vec!(1_f64, 0_f64)) == 0);
    assert!(network.classify(vec!(0_f64, 0_f64)) == 1);
    assert!(network.classify(vec!(1_f64, 1_f64)) == 1);
  }
}
