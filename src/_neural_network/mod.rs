mod layer;

use self::layer::Layer;
use rand::prelude::*;

pub struct Network {
  pub layers: Vec<Layer>
}

impl Network {

  pub fn train(&mut self, training_matrix: Vec<(u8, Vec<f64>)>) -> f64 {
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

}
