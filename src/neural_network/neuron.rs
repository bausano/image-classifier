pub struct Neuron {
  weights: Vec<f64>
}

impl Neuron {
  pub fn new(weights: Vec<f64>) -> Neuron {
    Neuron { weights }
  }

  pub fn impulse(&self, inputs: &Vec<f64>, bias: bool) -> f64 {
    let mut x = 0_f64;

    for i in 0..inputs.len() {
      x += inputs[i] * self.weights[i];
    }

    if bias {
      x += self.weights[self.weights.len() - 1];
    }

    0_f64.max(x)
  }
}
