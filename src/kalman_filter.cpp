#include "kalman_filter.h"
#include <iostream>
#include <cmath>
using Eigen::MatrixXd;
using Eigen::VectorXd;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict(float delta_T, float noise_ax, float noise_ay) {
  // Update the state transition matrix F according to the 
  // new elapsed Time is measured in seconds.
  F_(0, 2) = delta_T;
  F_(1, 3) = delta_T;

  // Update the process noise covariance matrix.
  float dt_2 = std::pow(delta_T, 2.0);
  float dt_3 = std::pow(delta_T, 3.0);
  float dt_4 = std::pow(delta_T, 4.0);

  Q_ << (dt_4/4)*noise_ax, 0, (dt_3/2)*noise_ax, 0,
        0, (dt_4/4)*noise_ay, 0, (dt_3/2)*noise_ay,
        (dt_3/2)*noise_ax, 0, dt_2*noise_ax, 0,
        0, (dt_3/2)*noise_ay, 0, dt_2*noise_ay;

  // Predict the state
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}


void KalmanFilter::UpdateLaser(const VectorXd &z,
                               const MatrixXd &R,
                               const MatrixXd &H) {
  // update the state by using Kalman Filter equations
  VectorXd z_pred = H * x_;
  VectorXd y = z - z_pred;
  Update(y, H, R);
}

void KalmanFilter::UpdateRadar(const VectorXd &z,
                               const MatrixXd &R,
                               const MatrixXd &H) {
  // update the state by using Extended Kalman Filter equations
  VectorXd z_pred = CartesianToPolar(x_);
  VectorXd y = z - z_pred;
  y[1] = NormalizePhi(y[1]);

  Update(y, H, R);
}

void KalmanFilter::Update(const VectorXd &y,
                          const MatrixXd &H,
                          const MatrixXd &R) {
  MatrixXd S = H * P_ * H.transpose() + R;
  MatrixXd K = P_ * H.transpose() * S.inverse();
  
  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H) * P_;
}

float KalmanFilter::NormalizePhi(float phi){
  // Normalize phi from -2pi to 2pi.
  float normalized_phi = phi;
  float pi2 = 2 * M_PI;

  if (phi > pi2) {
    normalized_phi = phi - pi2;
  } else if (phi < -pi2) {
    normalized_phi = phi + pi2;
  } 
  return normalized_phi;
}

VectorXd KalmanFilter::CartesianToPolar(const VectorXd &x) {
  VectorXd z_prev(3);
  float px = x[0];
  float py = x[1];
  float vx = x[2];
  float vy = x[3];

  z_prev[0] = std::sqrt(std::pow(px, 2.0) + std::pow(py, 2.0));
  z_prev[1] = atan2(py, px);
  z_prev[2] = (px*vx + py*vy)/z_prev[0];

  return z_prev;
}