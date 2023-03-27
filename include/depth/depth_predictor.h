#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>
#include <memory>
#include "ttc_object_avoidance/FeatAndDisplacement.h"

namespace depth
{

class DepthPredictor {
public:
    DepthPredictor(Eigen::Matrix3d body_to_camera, Eigen::Vector3d camera_offset);
    std::vector<Eigen::Vector3d> calculateDepth(Eigen::Vector3d body_velocity, Eigen::Quaterniond body_attitude, Eigen::Vector3d body_angular_vel, std::vector<ttc_object_avoidance::FeatAndDisplacement> feats);
private:
    Eigen::Matrix3d body_to_camera_;
    Eigen::Vector3d camera_offset_;
    float pxc, pyc, pzc;
};
}