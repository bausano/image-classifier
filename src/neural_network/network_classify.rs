use std::ops::Deref;
use super::network::Network;

impl Network {

  /// Classifies input (e.g. image) into one of the output layer categories.
  ///
  /// @param inputs Vector of same length as input layer
  /// @return Position of output neuron that fired the most
  pub fn classify (&self, inputs: Vec<f64>) -> u8 {
    // Serves as an iterator.
    let mut current_neuron: u8 = 0;
    // The neuron that fired the most.
    let mut strongest_neuron: Option<u8> = Some(current_neuron);
    // The intensity that the neuron fired with.
    let mut intensity: Option<f64> = None;

    // For each output neuron activation we compare the intensity.
    // This is basically a max function.
    for &probability in self.compute(inputs).iter() {
      println!("{:.2} % sure the answer is {}.", probability * 100_f64, current_neuron);
      match intensity {
        None => intensity = Some(probability),
        Some(x) => if x < probability {
          intensity = Some(probability);
          strongest_neuron = Some(current_neuron);
        }
      }

      current_neuron += 1;
    }

    // We use option here only to make the first strongest_neuron assigment
    // more comportable, hence can safely unwrap.
    strongest_neuron.unwrap()
  }

  /// Computes the activation of the network over given inputs.
  ///
  /// @param inputs Vector of same length as input layer
  /// @return Activation intensity of each neuron in output layer
  fn compute (&self, inputs: Vec<f64>) -> Vec<f64> {
    // We deference the pointer to the activation function.
    let activation_fn = self.activation.function.deref();

    // The prettiest line in the entire algorithm. We start with given inputs
    // and propagate the signal from layer to layer.
    self.layers.iter()
      .fold(inputs, |signal, layer| layer.activations(&signal, activation_fn))
  }

}
