extern crate rand;

pub mod reader;
pub mod neural_network;

use std::time::{SystemTime};
use neural_network::network::Network;
use neural_network::activation::Activation;

fn main() {
  // Bootstrap new network with randomly chosen weights.
  let mut network = Network::new(
    Activation::sigmoid(),
    vec!(64, 26, 10),
    12_f64,
  );

  // Trains the network on the training data.
  train_network(&mut network, 1000);

  // We read the digits from input file.
  let testing_data: Vec<(u8, Vec<f64>)> = reader::read_digits(
    include_str!("../data/cross_fold.txt"),
  ) .iter()
    .map(|digit| (digit.class, digit.grid.clone()))
    .collect();

  let success: usize = testing_data.iter()
    .fold(0, |success, (target, inputs)| {
      let succeded: bool = network.classify(inputs.clone()) == *target;

      if succeded { success + 1 } else { success }
    });

  println!("Correct {} out of {}.", success, testing_data.len());

  // TODO: Extract one input from the file and try to classify it.
}

fn train_network (network: &mut Network, iterations: usize) {
  // We read the digits from input file.
  let training_data: Vec<(u8, Vec<f64>)> = reader::read_digits(
    include_str!("../data/input.txt"),
  ) .iter()
    .map(|digit| (digit.class, digit.grid.clone()))
    .collect();
  let started_at = SystemTime::now();

  // Training the network.
  for _ in 0..iterations {
    network.train(&training_data);
  }

  println!(
    "Done {} iterations on {} samples in {:?}.",
    iterations,
    training_data.len(),
    SystemTime::now().duration_since(started_at).unwrap(),
  );
}

#[cfg(test)]
mod tests {
  // TODO: Cross validation.

  use super::neural_network::network::Network;
  use super::neural_network::activation::Activation;

  #[test]
  fn train_xor_gate() {
    let data = vec!(
      (0, vec!(1_f64, 1_f64)),
      (0, vec!(0_f64, 0_f64)),
      (1, vec!(1_f64, 0_f64)),
      (1, vec!(0_f64, 1_f64)),
    );

    let mut network: Network = Network::new(
      Activation::sigmoid(),
      vec!(2, 3, 2),
      3_f64,
    );

    for _ in 0..10000 {
      network.train(&data);
    }

    assert!(network.classify(vec!(0_f64, 1_f64)) == 1);
    assert!(network.classify(vec!(1_f64, 0_f64)) == 1);
    assert!(network.classify(vec!(0_f64, 0_f64)) == 0);
    assert!(network.classify(vec!(1_f64, 1_f64)) == 0);
  }
}
