use std::f64::consts::E;

pub struct Activation {

  // Desired activation function to map onto all layer outputs.
  pub function: Box<Fn(f64) -> f64>,

  // Derivative of the activation function.
  pub derivative: Box<Fn(f64) -> f64>,

}

impl Activation {

  // Sigmoid natural activation function that ranges the x to (-1;1).
  //
  // @return New Activation instance
  pub fn sigmoid () -> Self {
    Activation {
      function: Box::new(|x| 1_f64 / (1_f64 + E.powf(-x))),
      derivative: Box::new(|x| x * (1_f64 - x))
    }
  }

  // Leaky reluactivation function that scales down negative values only.
  //
  // @return New Activation instance
  pub fn leaky_relu () -> Self {
    Activation {
      function: Box::new(|x| (0.01_f64 * x).max(x)),
      derivative: Box::new(|x| {
        if x > 0_f64 {
          return 1_f64
        }

        0_f64
      })
    }
  }

}
