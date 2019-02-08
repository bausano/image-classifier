
// Network layer carries vector of neuron and performs computations on them.
pub struct Layer {

  // Vector of neurons with associated weights and bias weight. Each weight
  // is semantically connected to one neuron in previous layer. The bias weight
  // is always on maximum input, 1, therefore we don't have to keep a connection
  // to a bias neuron from previous layer.
  //
  // Following vector is therefore in following format:
  // neurons: Vector<(Neuron) bias, weights>
  //
  // At the end of the training process, we change the floats.
  pub neurons: Vec<(f64, Vec<f64>)>,

}

impl Layer {

  // Layer builder that is used to bootstrap the layer.
  //
  // @param neurons Vector of neurons in the layer
  // @return New layer instance
  pub fn from (neurons: Vec<(f64, Vec<f64>)>) -> Layer {
    Layer { neurons }
  }

  // Calculates the activations for each neuron against inputs. The network
  // builder ensures that there is going to be same number of weights as inputs.
  //
  // @param inputs Activations from previous layer
  // @param activation_function Function of x which puts the output in range
  // @return Activations for this layer
  pub fn activations (
    &self,
    inputs: &Vec<f64>,
    activation_function: &Fn(f64) -> f64,
  ) -> Vec<f64> {
    self.neurons.iter()
      .map(|neuron| {
        // Destruct each neuron into its weights and bias.
        let (bias, weights) = neuron;

        let mut product = 0_f64;
        // Matrix multiplication of weights and inputs.
        for i in 0..weights.len() {
          product += weights[i] * inputs[i];
        }

        // Bias has always input 1, therefore we can just add the bias weight.
        product + bias
      })
      // We map every neurons activation with given activation function.
      .map(activation_function)
      .collect()
  }
}
