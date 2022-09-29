#include "vision/gyro_integrator.h"

namespace vision
{

GyroIntegrator::GyroIntegrator(double t_delta) :
  t_delta_{t_delta}, bias_{Eigen::Vector3d::Zero()}, last_index_{0}
{}

void GyroIntegrator::insert_new_measurement(std::pair<double, Eigen::Vector3d> gyro_meas)
{
  measurements_.push_back(gyro_meas);
}

void GyroIntegrator::update_bias(Eigen::Vector3d bias)
{
  bias_ = bias;
}

lie_groups::SO3d GyroIntegrator::update(double image_time)
{
  // this approximates relative rotation, assumes number of measurements integrated, time steps, and ang_vel are low
  Eigen::Vector3d sum = Eigen::Vector3d::Zero();
  while(measurements_[++last_index_].first - t_delta_ < image_time)
  {
    sum += (measurements_[last_index_].second - bias_) * (measurements_[last_index_].first - measurements_[last_index_-1].first);
  }
  last_index_--;

  R_integrated_ = lie_groups::SO3d::Exp(sum) * R_integrated_; 
  return R_integrated_;
}

void GyroIntegrator::reset(double image_time)
{
  R_integrated_ = lie_groups::SO3d();
  while(measurements_[0].first - t_delta_ < image_time) measurements_.erase(measurements_.begin());
  last_index_ = 0;
}

} // namespace vision