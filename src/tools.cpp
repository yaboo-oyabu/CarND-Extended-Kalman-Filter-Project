#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::ArrayXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * Calculate the RMSE.
   */
  ArrayXd residual(4);
  ArrayXd sqsum(4);
  ArrayXd mean(4);
  VectorXd rmse(4);

  sqsum << 0,0,0,0;
  rmse << 0,0,0,0;

  if (estimations.size() == 0) {
    throw std::length_error("the estimation vector size should not be zero");
  } else if (estimations.size() != ground_truth.size()) {
    throw std::length_error(
      "the estimation vector size should equal ground truth vector size");
  }

  for (int i=0; i < estimations.size(); ++i) {
    residual = (estimations[i] - ground_truth[i]).array();
    sqsum += residual * residual;
  }

  rmse = (sqsum/estimations.size()).sqrt().matrix();  
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
   * Calculate a Jacobian.
   */

  MatrixXd Hj(3,4);
  
  // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  float d1 = px*px + py*py;
  float d2 = std::pow(d1, 0.5);
  float d3 = std::pow(d1, 1.5);
  
  // check division by zero
  if (px == 0 & py == 0){
    throw std::overflow_error("Divide by zero exception");
  }
  
  // compute the Jacobian matrix
  Hj << px/d2, py/d2, 0, 0,
        -py/d1, px/d1, 0, 0,
        py*(vx*py-vy*px)/d3, px*(vy*px-vx*py)/d3, px/d2, py/d2;

  return Hj;
}
