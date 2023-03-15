#include "depth/depth_predictor.h"

namespace depth
{

DepthPredictor::DepthPredictor(Eigen::Matrix3f body_to_camera, Eigen::Vector3f camera_offset):body_to_camera_(body_to_camera),camera_offset_(camera_offset)
{}

std::vector<DepthInfo> DepthPredictor::calculateDepth(Eigen::Vector3f velocity, Eigen::Quaternionf quat, Eigen::Vector3f angular_vel, std::vector<FeatInfo> feats)
{
    Eigen::Matrix3f rot_mat(quat);
    Eigen::Vector3f velocity_camera = body_to_camera_ * rot_mat.transpose() * velocity;
    Eigen::Vector3f omega_camera = body_to_camera_ * angular_vel;
    float velx, vely, velz, wx, wy, wz;
    velx = velocity_camera(0);
    vely = velocity_camera(1);
    velz = velocity_camera(2);

    wx = omega_camera(0);
    wy = omega_camera(1);
    wz = omega_camera(2);

    return std::vector<DepthInfo>();
}

}