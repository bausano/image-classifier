//mod reader;
mod neural_network;

//use reader::digit::Digit;
use neural_network::Network;

fn main() {
  //let digits: Vec<Digit> = reader::read_digits();

  //println!("Analyzing {} digits.", digits.len());
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
