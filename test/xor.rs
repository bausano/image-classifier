/**
 * let network: Network = Network::from(vec!(
    vec!(
      vec!(0.1_f64, 0.1_f64, -0.1_f64), vec!(0.9_f64, 0.9_f64, 0_f64)
    ),
    vec!(
      vec!(-10_f64, 1_f64, -0.01_f64), vec!(100_f64, -1_f64, 0.01_f64)
    )
  ));

  // for 0_f64.max(x)
 */

/**
 *
  assert!(network.classify(vec!(0_f64, 1_f64)) == 0);
  assert!(network.classify(vec!(1_f64, 0_f64)) == 0);
  assert!(network.classify(vec!(0_f64, 0_f64)) == 1);
  assert!(network.classify(vec!(1_f64, 1_f64)) == 1);
 */
