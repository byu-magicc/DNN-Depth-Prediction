#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>

namespace depth
{

struct DepthInfo {
    Eigen::Vector2f point;
    float depth;
};

struct FeatInfo {
    Eigen::Vector2f position;
    Eigen::Vector2f velocity;
};

class DepthPredictor {
public:
DepthPredictor(Eigen::Matrix3f body_to_camera, Eigen::Vector3f camera_offset);
std::vector<DepthInfo> calculateDepth(Eigen::Vector3f velocity, Eigen::Quaternionf quat, Eigen::Vector3f angular_vel, std::vector<FeatInfo> feats);
private:
Eigen::Matrix3f body_to_camera_;
Eigen::Vector3f camera_offset_;
};
}