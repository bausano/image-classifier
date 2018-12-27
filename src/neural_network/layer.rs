use neural_network::neuron::Neuron;

pub struct Layer {
  neurons: Vec<Neuron>,
  bias: bool
}

impl Layer {
  pub fn from(neuron_weights: Vec<Vec<f64>>) -> Layer {
    Layer {
      bias: true,
      neurons: neuron_weights
        .into_iter()
        .map(|weights| Neuron::new(weights))
        .collect()
    }
  }

  pub fn process(&self, inputs: Vec<f64>) -> Vec<f64> {
    self.neurons.iter()
      .map(|neuron| neuron.impulse(&inputs, self.bias))
      .collect()
  }
}
