extern crate rand;

pub mod reader;
pub mod neural_network;

use neural_network::network::Network;
use std::time::{SystemTime, Duration};
use neural_network::activation::Activation;

fn main() {

  // Bootstrap new network with randomly chosen weights.
  let mut network = Network::new(
    Activation::sigmoid(),
    vec!(64, 128, 128, 10)
  );

  network.batch_size = 10;
  network.max_lr = 0.8_f64;
  network.min_lr = 0.1_f64;
  network.step_size = 8_f64;

  // How many times should the training data be processed.
  let iterations = 6 * (network.step_size.floor() as usize * 2) + network.step_size.floor() as usize + 1;

  // Trains the network on the training data.
  let (duration, samples) = train_network(&mut network, iterations);

  // We read the digits from input file.
  let (success, total) = validate_network(&network);

  println!(
    "Done {} iterations on {} samples in {:?}.", iterations, samples, duration,
  );

  println!("Correct {} out of {}.", success, total);
}

/// Trains the network on the input testing data.
/// TODO: Dynamicaly load the input file.
///
/// @param network Network instance to train
/// @param iterations How many times should be the data set processed
/// @return Tuple in format (duration_of_training, samples_in_per_data_set)
fn train_network (network: &mut Network, iterations: usize) -> (Duration, usize) {
  // We read the digits from input file.
  let training_data: Vec<(u8, Vec<f64>)> = reader::read_digits(
    include_str!("../data/input.txt"),
  ) .iter()
    .map(|digit| (digit.class, digit.grid.clone()))
    .collect();

  let started_at = SystemTime::now();

  // Training the network.
  for epoch in 0..iterations {
    network.train(&training_data, epoch);
  }

  (
    SystemTime::now().duration_since(started_at).unwrap(),
    training_data.len(),
  )
}

/// Performs a cross fold validation on the training set.
///
/// @param network Trained network
/// @return Tuple in format (successful_classifications, samples)
fn validate_network (network: &Network) -> (usize, usize) {
  // We read the digits from input file.
  let testing_data: Vec<(u8, Vec<f64>)> = reader::read_digits(
    include_str!("../data/cross_fold.txt"),
  ) .iter()
    .map(|digit| (digit.class, digit.grid.clone()))
    .collect();

  // Calculate successful attemps over the cross fold data set.
  let success: usize = testing_data.iter()
    .fold(0, |success, (target, inputs)| {
      let succeded: bool = network.classify(inputs.clone()) == *target;

      if succeded { success + 1 } else { success }
    });

  (success, testing_data.len())
}

#[cfg(test)]
mod tests {
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
    );

    network.learning_rate = 3_f64;
    network.batch_size = 3;

    for epoch in 0..10000 {
      network.train(&data, epoch);
    }

    assert!(network.classify(vec!(0_f64, 1_f64)) == 1);
    assert!(network.classify(vec!(1_f64, 0_f64)) == 1);
    assert!(network.classify(vec!(0_f64, 0_f64)) == 0);
    assert!(network.classify(vec!(1_f64, 1_f64)) == 0);
  }
}
