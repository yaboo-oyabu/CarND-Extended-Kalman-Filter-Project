#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement function matrix - laser
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  // will be overwritten in the first measurement.
  ekf_.x_ = VectorXd(4);
  ekf_.x_ << 1, 1, 1, 1;
  
  // initial uncertainty covariance.
  ekf_.P_ = MatrixXd(4,4);
  ekf_.P_ << 1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1000, 0,
             0, 0, 0, 1000;

  // the initial transition matrix F_
  ekf_.F_ = MatrixXd(4,4);
  ekf_.F_ << 1, 0, 1, 0,
             0, 1, 0, 1,
             0, 0, 1, 0,
             0, 0, 0, 1;

  // process noise.
  ekf_.Q_ = MatrixXd(4,4);
  ekf_.Q_ << 0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0;

  // set the acceleration noise components
  noise_ax = 9;
  noise_ay = 9;
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) {
    /**
     * Initialize the state ekf_.x_ with the first measurement, and
     * create the covariance matrix.
     */

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates and
      // initialize state.
      ekf_.x_ = PolarToCartesian(measurement_pack.raw_measurements_);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // Initialize state.
      ekf_.x_ << measurement_pack.raw_measurements_[0], 
                 measurement_pack.raw_measurements_[1], 
                 0,
                 0; 
           
    }
    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /**
   * Prediction
   */

  float delta_T = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0; 
  ekf_.Predict(delta_T, noise_ax, noise_ay);

  /**
   * Update
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    Hj_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.UpdateRadar(measurement_pack.raw_measurements_, R_radar_, Hj_);
  } else {
    // Laser updates
    ekf_.UpdateLaser(measurement_pack.raw_measurements_, R_laser_, H_laser_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;

  previous_timestamp_ = measurement_pack.timestamp_;
}

VectorXd FusionEKF::PolarToCartesian(const VectorXd &measurements) {
  VectorXd out(4);
  float rho = measurements[0];
  float phi = measurements[1];
  float rho_dot = measurements[2];

  out[0] = rho * sin(phi); // px
  out[1] = rho * cos(phi); // py
  out[2] = rho_dot * sin(phi); // vx
  out[3] = rho_dot * cos(phi); // vy
  return out;
}
