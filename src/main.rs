extern crate rand;

mod reader;
pub mod neural_network;

use reader::digit::Digit;
use neural_network::network::Network;
use neural_network::activation::Activation;

fn main() {
  // We read the digits from input file.
  let digits: Vec<Digit> = reader::read_digits();

  // Destruct the digits into the training data.
  let training_data: Vec<(u8, Vec<f64>)> = digits.iter()
    .map(|digit| (digit.class, digit.grid.clone()))
    .collect();

  // Bootstrap new network with randomly chosen weights.
  let mut network = Network::new(
    Activation::leaky_relu(),
    vec!(64, 12, 12, 10),
  );

  // Training the network.
  network.train(training_data);

  // TODO: Extract one input from the file and try to classify it.
}

#[cfg(test)]
mod tests {
  // TODO: Cross validation.

  use super::neural_network::network::Network;
  use super::neural_network::activation::Activation;

  #[test]
  fn xor_gate() {
    let schema = vec!(
      vec!(
        (-0.1_f64, vec!(0.1_f64, 0.1_f64)), (0_f64, vec!(0.9_f64, 0.9_f64))
      ),
      vec!(
        (-0.01_f64, vec!(-10_f64, 1_f64)), (0.01_f64, vec!(100_f64, -1_f64))
      )
    );

    let network: Network = Network::from(
      Activation::leaky_relu(),
      schema,
    );

    assert!(network.classify(vec!(0_f64, 1_f64)) == 0);
    assert!(network.classify(vec!(1_f64, 0_f64)) == 0);
    assert!(network.classify(vec!(0_f64, 0_f64)) == 1);
    assert!(network.classify(vec!(1_f64, 1_f64)) == 1);
  }

  #[test]
  fn train_xor_gate() {
    let mut network: Network = Network::new(
      Activation::leaky_relu(),
      vec!(2, 2, 2),
    );

    network.train(vec!(
      (0, vec!(1_f64, 1_f64)),
      (0, vec!(0_f64, 0_f64)),
      (1, vec!(1_f64, 0_f64)),
      (1, vec!(0_f64, 1_f64))
    ));

    println!("\nWeights");
    for layer in network.layers.iter() {
      for (bias, weights) in layer.neurons.iter() {
        println!("{:.2?} + {}", weights, bias);
      }

      println!("\n");
    }

    assert!(network.classify(vec!(0_f64, 1_f64)) == 0);
    assert!(network.classify(vec!(1_f64, 0_f64)) == 0);
    assert!(network.classify(vec!(0_f64, 0_f64)) == 1);
    assert!(network.classify(vec!(1_f64, 1_f64)) == 1);
  }
}
