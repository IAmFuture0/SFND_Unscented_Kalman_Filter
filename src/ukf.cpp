#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;


#include <iostream>
/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);
  x_.fill(0);
  
  // initial covariance matrix
  P_ = MatrixXd(5, 5);
  P_.fill(0);
  
  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3; // 30

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1; // 30
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  n_x_ = 5;
  n_z_rad_ = 3;
  n_z_lid_ = 2;
  lambda_ = 3 - n_x_;
  
  x_pred_ = VectorXd(n_x_);
  x_pred_.fill(0);
  
  P_pred_ = MatrixXd(n_x_, n_x_);
  P_pred_.fill(0);
  
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  Xsig_pred_.fill(0);
  
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_.fill(0);
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  
  // initialization ukf state with first lidar measurement
  if((!is_initialized_) && meas_package.sensor_type_ == MeasurementPackage::LASER){
    x_(0) = meas_package.raw_measurements_(0);
    x_(1) = meas_package.raw_measurements_(1);
    P_(0, 0) = 1; // 0.225
    P_(1, 1) = 1; // 0.225
    P_(2, 2) = 1;
    P_(3, 3) = 1;
    P_(4, 4) = 1; 

    is_initialized_ = true;
  } else {
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) UpdateRadar(meas_package);
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) UpdateLidar(meas_package);
  }
  
  return;
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  
  // Generate Sigma Points (UKF Augmentation)
  // - create augmented state
  VectorXd x_aug_ = VectorXd(n_aug_); 
  x_aug_.fill(0.0);
  x_aug_.head(5) = x_;
  x_aug_(5) = 0;
  x_aug_(6) = 0;

  // - create augmented covariance
  MatrixXd P_aug_ = MatrixXd(n_aug_, n_aug_);
  P_aug_.fill(0.0);
  P_aug_.topLeftCorner(5, 5) = P_;
  P_aug_(5, 5) = std_a_ * std_a_;
  P_aug_(6, 6) = std_yawdd_ * std_yawdd_;

  // - create square root matrix
  MatrixXd L_aug = P_aug_.llt().matrixL();

  // - create augmented sigma points
  MatrixXd Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug_.fill(0.0);
  Xsig_aug_.col(0) = x_aug_;
  for(int i = 0; i < n_aug_; i++){
    Xsig_aug_.col(i+1)        = x_aug_ + sqrt(lambda_ + n_aug_) * L_aug.col(i) ;
    Xsig_aug_.col(i+n_aug_+1) = x_aug_ - sqrt(lambda_ + n_aug_) * L_aug.col(i);
  }

  // Predict Sigma Points
  for(int i = 0; i < 2 * n_aug_ + 1; i++){
    // - extract values for better readability
    double p_x = Xsig_aug_(0, i);
    double p_y = Xsig_aug_(1, i);
    double v = Xsig_aug_(2, i);
    double yaw = Xsig_aug_(3, i);
    double yawd = Xsig_aug_(4, i);
    double nu_a = Xsig_aug_(5, i);
    double nu_yawdd = Xsig_aug_(6, i);

    // - predicted state values
    double px_p, py_p, v_p, yaw_p, yawd_p;

    // avoid division by zero
    if(fabs(yawd) > 0.001){
      px_p = p_x + (v / yawd) * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + (v / yawd) * (-1 * cos(yaw + yawd * delta_t) + cos(yaw));
    } else {
      px_p = p_x + v * cos(yaw) * delta_t;
      py_p = p_y + v * sin(yaw) * delta_t;
    }
    v_p = v;
    yaw_p = yaw + yawd * delta_t;
    yawd_p = yawd;

    // - add noise
    px_p = px_p + 0.5 * pow(delta_t, 2) * nu_a * cos(yaw);
    py_p = py_p + 0.5 * pow(delta_t, 2) * nu_a * sin(yaw);
    v_p = v_p + delta_t * nu_a;
    yaw_p = yaw_p + 0.5 * pow(delta_t, 2) * nu_yawdd;
    yawd_p = yawd_p + delta_t * nu_yawdd;

    // - write predicted sigma point into right column
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }

  // Predict Mean and Covariance
  // - set weights
  double weight_0 = lambda_ / (lambda_ + n_aug_);
  weights_(0) = weight_0;
  for(int i = 0; i < 2 * n_aug_ + 1; i++){ // 2n+1 weights
    double weight = 0.5 / (lambda_ + n_aug_);
    weights_(i) = weight;
  }

  // - predicted state mean
  for(int i = 0; i < 2 * n_aug_ + 1; i++){ // iterate over sigma points
    x_pred_ = x_pred_ + weights_(i) * Xsig_pred_.col(i);
  }

  // - predicted state covariance matrix
  for(int i = 0; i < 2 * n_aug_ + 1; i++){ // iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_pred_;
    // angle normalization
    while(x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
    while(x_diff(3) < (-1) * M_PI) x_diff(3) += 2. * M_PI;
    P_pred_ = P_pred_ + weights_(i) * x_diff * x_diff.transpose();
  }
  
  x_ = x_pred_;
  P_ = P_pred_;
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  // Predict Measurement
  // - transform sigma points into measurement space
  
  MatrixXd Zsig = MatrixXd(n_z_lid_, 2 * n_aug_ + 1);
  Zsig.fill(0.0);
  
  for(int i = 0; i < 2 * n_aug_ + 1; i++){
    // extract values for better readability
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);

    // measurement model
    Zsig(0, i) = p_x;    // px
    Zsig(1, i) = p_y;    // py
  }
  
  // - mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_lid_);
  z_pred.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++){
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }
  
  // - innovation covariance matrix S
  MatrixXd S = MatrixXd(n_z_lid_, n_z_lid_);
  S.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++){ // 2n+1 sigma points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // - add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z_lid_, n_z_lid_);
  R << std_laspx_*std_laspx_, 0,
       0, std_laspy_*std_laspy_;
  S = S + R;
  
  // Update State
  // create example vector for incoming lidar measurement
  VectorXd z = VectorXd(n_z_lid_);
  z << meas_package.raw_measurements_(0),
       meas_package.raw_measurements_(1);

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z_lid_);
  
  // calculate cross correlation matrix
  Tc.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++){ // 2n+1 sigma points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_pred_;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }  
  
  // Kalman gian K
  MatrixXd K = Tc * S.inverse();

  // residual
  VectorXd z_diff = z - z_pred;
  
  // angle normalization
  while(z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
  while(z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

  // update state mean and covariance matrix
  x_ = x_pred_ + K * z_diff;
  P_ = P_pred_ - K * S * K.transpose();  
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

  // Predict Measurement
  // - transform sigma points into measurement space
  MatrixXd Zsig = MatrixXd(n_z_rad_, 2 * n_aug_ + 1);
  Zsig.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++){
    // extract values for better readability
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);
    
    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    // measurement model
    Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);                         // r
    Zsig(1, i) = atan2(p_y, p_x);                                     // phi
    Zsig(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y);// r_dot
  }
  
  // - mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_rad_);
  z_pred.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++){
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // - innovation covariance matrix S
  MatrixXd S = MatrixXd(n_z_rad_, n_z_rad_);
  S.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++){ // 2n+1 sigma points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // - add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z_rad_, n_z_rad_);
  R << std_radr_*std_radr_ , 0, 0,
       0, std_radphi_*std_radphi_, 0,
       0, 0, std_radrd_*std_radrd_;
  S = S + R;

  // Update State
  // create example vector for incoming radar measurement
  VectorXd z = VectorXd(n_z_rad_);
  z << meas_package.raw_measurements_(0),
       meas_package.raw_measurements_(1),
       meas_package.raw_measurements_(2);

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z_rad_);

  // calculate cross correlation matrix
  Tc.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++){ // 2n+1 sigma points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    while(z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while(z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_pred_;
    // angle normalization
    while(x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
    while(x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }  

  // Kalman gian K
  MatrixXd K = Tc * S.inverse();

  // residual
  VectorXd z_diff = z - z_pred;

  // angle normalization
  while(z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
  while(z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

  // update state mean and covariance matrix
  x_ = x_pred_ + K * z_diff;
  P_ = P_pred_ - K * S * K.transpose();
}
