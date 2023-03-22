#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>
#include "ttc_object_avoidance/FeatAndDisplacement.h"
#include "ttc_object_avoidance/FeatAndDepth.h"

namespace depth
{

// struct DepthInfo {
//     Eigen::Vector2f point;
//     float depth;
//     DepthInfo(Eigen::Vector2f point, float depth):point(point), depth(depth) {}
// };

// struct FeatInfo {
//     Eigen::Vector2f position;
//     Eigen::Vector2f velocity;
// };

class DepthPredictor {
public:
    DepthPredictor(Eigen::Matrix3d body_to_camera, Eigen::Vector3d camera_offset);
    std::vector<ttc_object_avoidance::FeatAndDepth> calculateDepth(Eigen::Vector3d velocity, Eigen::Quaterniond quat, Eigen::Vector3d angular_vel, std::vector<ttc_object_avoidance::FeatAndDisplacement> feats);
private:
    Eigen::Matrix3d body_to_camera_;
    Eigen::Vector3d camera_offset_;
    float pxc, pyc, pzc;
};
}