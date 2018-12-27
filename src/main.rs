extern crate rand;

mod reader;
mod neural_network;

use reader::digit::Digit;
use neural_network::Network;

fn main() {
  let mut digits: Vec<Digit> = reader::read_digits();
  let digit = digits.pop().unwrap();

  let network = Network::new(vec!(64, 12, 12, 10));

  println!("Network classify {}", network.classify(digit.grid.clone()));
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
