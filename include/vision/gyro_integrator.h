#pragma once

#include <vector>
#include <Eigen/Dense>

#include "lie_groups/so3.hpp"

namespace vision
{

class GyroIntegrator
{
public:
  GyroIntegrator(double t_delta);

  void insert_new_measurement(std::pair<double, Eigen::Vector3d> gyro_meas);
  void update_bias(Eigen::Vector3d bias);

  lie_groups::SO3d update(double image_time);
  void reset(double image_time);

private:
  std::vector<std::pair<double, Eigen::Vector3d>> measurements_; // assumed to be in ascending time order
  Eigen::Vector3d bias_;
  lie_groups::SO3d R_integrated_;
  int last_index_;

  double t_delta_; // time offset between camera and imu (t_imu = t_camera + t_delta_)
};

} // namespace vision