#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  is_initialized_ = false;

  time_us_ = 0;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd::Zero(5);

  // initial covariance matrix
  P_ = MatrixXd::Identity(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;

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

  // state dimension
  n_x_ = x_.size();

  // augmented dimension: (state dimension + acceleration noise nu_a + yaw
  // acceleration nu_yawdd)
  n_aug_ = x_.size() + 2;

  n_sig_ =  2 * n_aug_ + 1;

  // spreading parameter
  lambda_ = 3 - n_aug_;

  R_lidar_ = MatrixXd(2, 2);
  R_lidar_ << std_laspx_ * std_laspx_, 0,
    0, std_laspy_ * std_laspy_;

  R_radar_ = MatrixXd(3, 3);
  R_radar_ << std_radr_ * std_radr_, 0, 0,
    0, std_radphi_ * std_radphi_, 0,
    0, 0, std_radrd_ * std_radrd_;

  // create predicted sigma points matrix
  Xsig_pred_ = MatrixXd::Zero(n_x_, n_sig_);

  // set weights
  weights_ = VectorXd(n_sig_);
  weights_.fill(0.5 / (lambda_ + n_aug_));
  weights_(0) = lambda_ / (lambda_ + n_aug_);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage& meas_package) {
  if (!is_initialized_) {
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1],
      0, 0, 0;
    } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      double rho = meas_package.raw_measurements_[0];
      double phi = meas_package.raw_measurements_[1];
      x_ << rho * cos(phi), rho * sin(phi), 0, 0, 0;
    }

    P_(0, 0) = 0.05;
    P_(1, 1) = 0.05;

    time_us_        = meas_package.timestamp_;
    is_initialized_ = true;
    std::cout << "UKF initialized" << std::endl;
    return;
  }

  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;

  if (dt > 0.001) {
    time_us_ = meas_package.timestamp_;
    Prediction(dt);
  }

  if ((meas_package.sensor_type_ == MeasurementPackage::LASER) && use_laser_) {
    UpdateLidar(meas_package);
  } else if ((meas_package.sensor_type_ == MeasurementPackage::RADAR) &&
             use_radar_) {
    UpdateRadar(meas_package);
  }

  // std::cout << P_ << std::endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  // Step 1:  Generate Augmented Sigma points

  // create augmented mean vector
  VectorXd x_aug = VectorXd::Zero(n_aug_);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);

  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd::Zero(n_aug_, n_sig_);

  // set augmented mean state
  // mean for nu_a and nu_yawdd are both zeros.
  x_aug.head(5) = x_;

  // set augmented covariance matrix
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug(5, 5)               = std_a_ * std_a_;
  P_aug(6, 6)               = std_yawdd_ * std_yawdd_;

  // create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  // set augmented sigma points
  Xsig_aug.col(0) = x_aug;

  for (int i = 0; i < n_aug_; i++) {
    Xsig_aug.col(i + 1)          = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }

  // Step 2: Predict sigma points

  double px, py, v, yaw, yawd, nu_a, nu_yawdd;
  double px_p, py_p, v_p, yaw_p, yawd_p, nu_a_p, nu_yawdd_p;

  for (int i = 0; i < n_sig_; i++) {
    // for better readability
    px       = Xsig_aug(0, i);
    py       = Xsig_aug(1, i);
    v        = Xsig_aug(2, i);
    yaw      = Xsig_aug(3, i);
    yawd     = Xsig_aug(4, i);
    nu_a     = Xsig_aug(5, i);
    nu_yawdd = Xsig_aug(6, i);

    // avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = px + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = py + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    } else {
      px_p = px + v * delta_t * cos(yaw);
      py_p = py + v * delta_t * sin(yaw);
    }

    // add noise for position states
    px_p += 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p += 0.5 * nu_a * delta_t * delta_t * sin(yaw);

    // predict the rest states
    v_p    = v + nu_a * delta_t;
    yaw_p  = yaw + yawd * delta_t + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd + nu_yawdd * delta_t;

    // write predited sigma point in right column
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }

  // Step 3: Compute predited mean and covariance matrix

  // predict state mean from predicted sigma points
  x_ = Xsig_pred_ * weights_;

  // reset state covariance matrix and predict it from predicted state mean and
  // its distance to predicted sigma points.
  P_.fill(0.0);
  VectorXd x_diff = VectorXd::Zero(n_x_);

  for (int i = 0; i < (n_sig_); i++) {
    x_diff = Xsig_pred_.col(i) - x_;

    // angle normalization for yaw
    x_diff(3) = atan2(sin(x_diff(3)), cos(x_diff(3)));

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage& meas_package) {
  static int num_of_measurements = 0;

  // total number of estimations which pass chi_square_mark
  static int num_of_bad_estimations = 0;

  // chi_square distribution of df = 2 and p = 0.05
  static const double chi_square_mark = 5.99;

  int n_z = 2;

  // parse out real measurement from meas_package
  VectorXd z = meas_package.raw_measurements_;

  // Step 1: create and set matrix for sigma points in measurement space

  MatrixXd Zsig = Xsig_pred_.block(0, 0, n_z, n_sig_);

  // Step 2: computed mean of sigma point in measurement space and corresponding
  // measurement covariance matrix

  // mean of sigma points in measurement space
  VectorXd z_pred = VectorXd::Zero(n_z);
  z_pred = Zsig * weights_;

  // measurement covariance matrix S
  MatrixXd S      = MatrixXd::Zero(n_z, n_z);
  VectorXd z_diff = VectorXd::Zero(n_z);

  for (int i = 0; i < n_sig_; i++) { // 2n+1 simga points
    // residual
    z_diff = Zsig.col(i) - z_pred;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // add measurement noise covariance matrix
  S = S + R_lidar_;

  // Step 3: measurement update

  // create correclation matrix Tc
  MatrixXd Tc     = MatrixXd::Zero(n_x_, n_z);
  VectorXd x_diff = VectorXd(n_x_);

  for (int i = 0; i < n_sig_; i++) { // 2n+1 simga points
    // residual
    z_diff = Zsig.col(i) - z_pred;

    // state difference
    x_diff = Xsig_pred_.col(i) - x_;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K
  MatrixXd K = Tc * S.inverse();

  // residual in between real measurement and predicted measurement
  z_diff = z - z_pred;

  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  // Step 4: Calculate Normalized Innovation Squared (NIS)
  double epsilon = z_diff.transpose() * S.inverse() * z_diff;
  ++num_of_measurements;

  if (epsilon >= chi_square_mark) {
    ++num_of_bad_estimations;
  }
  std::cout << "Lidar chi-square test: " << (double)num_of_bad_estimations /
  num_of_measurements << std::endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage& meas_package) {
  static int num_of_measurements = 0;

  // total number of estimations which pass chi_square_mark
  static int num_of_bad_estimations = 0;

  // chi_square distribution of df = 3 and p = 0.05
  static const double chi_square_mark = 7.82;

  // set measurement dimension: r, phi, and r_dot
  int n_z = 3;

  // parse out real measurement from meas_package
  VectorXd z = meas_package.raw_measurements_;

  // Step 1: create and set matrix for sigma points in measurement space

  MatrixXd Zsig = MatrixXd(n_z, n_sig_);

  // transform sigma points into measurement space
  double p_x, p_y, v, yaw, v1, v2;
  double c0 = 0;

  for (int i = 0; i < n_sig_; i++) { // 2n+1 simga points
    // extract values for better readibility
    p_x = Xsig_pred_(0, i);
    p_y = Xsig_pred_(1, i);
    v   = Xsig_pred_(2, i);
    yaw = Xsig_pred_(3, i);

    v1 = cos(yaw) * v;
    v2 = sin(yaw) * v;

    c0 = sqrt(p_x * p_x + p_y * p_y);

    // measurement model
    if (c0 < 0.001) {
      Zsig(0, i) = 0;
      Zsig(1, i) = 0;
      Zsig(2, i) = 0;
    } else {
      Zsig(0, i) = c0;                         // r
      Zsig(1, i) = atan2(p_y, p_x);            // phi
      Zsig(2, i) = (p_x * v1 + p_y * v2) / c0; // r_dot
    }
  }

  // Step 2: computed mean of sigma point in measurement space and corresponding
  // measurement covariance matrix

  // mean of sigma points in measurement space
  VectorXd z_pred = VectorXd::Zero(n_z);
  z_pred = Zsig * weights_;

  // measurement covariance matrix S
  MatrixXd S      = MatrixXd::Zero(n_z, n_z);
  VectorXd z_diff = VectorXd::Zero(n_z);

  for (int i = 0; i < n_sig_; i++) { // 2n+1 simga points
    // residual
    z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    z_diff(1) = atan2(sin(z_diff(1)), cos(z_diff(1)));

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // add measurement noise covariance matrix
  S = S + R_radar_;

  // Step 3: measurement update

  // create correclation matrix Tc
  MatrixXd Tc     = MatrixXd::Zero(n_x_, n_z);
  VectorXd x_diff = VectorXd(n_x_);

  for (int i = 0; i < n_sig_; i++) { // 2n+1 simga points
    // residual
    z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    z_diff(1) = atan2(sin(z_diff(1)), cos(z_diff(1)));

    // state difference
    x_diff = Xsig_pred_.col(i) - x_;

    // angle normalization
    x_diff(3) = atan2(sin(x_diff(3)), cos(x_diff(3)));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K
  MatrixXd K = Tc * S.inverse();

  // residual in between real measurement and predicted measurement
  z_diff = z - z_pred;

  // angle normalization
  z_diff(1) = atan2(sin(z_diff(1)), cos(z_diff(1)));

  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  // Step 4: Calculate Normalized Innovation Squared (NIS)
  double epsilon = z_diff.transpose() * S.inverse() * z_diff;
  ++num_of_measurements;

  if (epsilon >= chi_square_mark) {
    ++num_of_bad_estimations;
  }
  std::cout << "Radar chi-square test: " << (double)num_of_bad_estimations /
  num_of_measurements << std::endl;
}
