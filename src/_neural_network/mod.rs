mod layer;

use self::layer::Layer;
use rand::prelude::*;

pub struct Network {
  pub layers: Vec<Layer>
}

impl Network {

  pub fn train(&mut self, training_matrix: Vec<(u8, Vec<f64>)>) -> f64 {
    let activation_matrix = self.calculate_activation_matrix(&training_matrix);

    let errors: Vec<Vec<f64>> = activation_matrix.iter().enumerate()
      .map(|(i, digit)| {
        let outputs: &Vec<f64> = digit.last().unwrap();
        let (target, _) = training_matrix[i];
        let target = usize::from(target.clone());

        outputs.iter().enumerate()
          .map(|(k, o)| o - (if target == k { 1_f64 } else { 0_f64 }))
          .map(|x| x.powi(2) * 0.5_f64)
          .collect()
      })
      .collect();

    // learning_rate = 0.5
    // for each digit
      // error_total = get total error
      // for each output neuron
        // activation_der = change of activation fn with respect to input (activation * (1 - activation))
        // w_change = learning_rate * (error_tltal * activation_der * activation_from_prev_neuron)

    println!("Cost of training examples is {}.", errors.iter().fold(0_f64, |sum, x| sum + x.iter().fold(0_f64, |y, f| y + f)));

    let error_matrix = self.calculate_error_matrix(&errors);

    for (activation_layer, error_layer) in activation_matrix.iter().zip(error_matrix.iter()) {
      println!("--- Layer --- \n");

      println!("Activation Matrix");
      for layer in activation_layer.iter() {
        for cell in layer.iter() {
          print!("{:.2} ", cell);
        }
        println!("");
      }

      println!("\nError Matrix");
      for layer in error_layer.iter() {
        for cell in layer.iter() {
          print!("{:.2} ", cell);
        }
        println!("");
      }
    }

    0_f64
  }

  fn calculate_activation_matrix(&self, matrix: &Vec<(u8, Vec<f64>)>)
    -> Vec<Vec<Vec<f64>>> {
    matrix.iter()
      .map(|(_class, inputs)| self.layers.iter().fold(
        vec!(inputs.clone()),
        |mut activations, layer| {
          let output = layer.process(&activations.last().unwrap());

          activations.push(output);

          activations
      }))
      .collect()
  }

  fn calculate_error_matrix(&self, matrix: &Vec<Vec<f64>>)
    -> Vec<Vec<Vec<f64>>> {
    matrix.iter().map(|errors| self.layers.iter().rev()
      .fold(vec!(errors.clone()), |mut carry, layer| {
        let item = layer.neurons.iter()
          .map(|(bias, weights)| {
            let previous_errors: &Vec<f64> = carry.last().unwrap();

            let product = weights.iter().zip(previous_errors.iter())
              .fold(0_f64, |sum, (weight, error)| sum + weight * error);

            product + previous_errors.iter().fold(0_f64, |sum, e| sum + e * bias)
          })
          .collect::<Vec<f64>>();

        carry.push(item);

        carry
      })
    ).collect::<Vec<Vec<Vec<f64>>>>()
  }
}