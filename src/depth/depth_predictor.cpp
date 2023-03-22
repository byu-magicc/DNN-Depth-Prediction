#include "depth/depth_predictor.h"

namespace depth
{

DepthPredictor::DepthPredictor(Eigen::Matrix3d body_to_camera, Eigen::Vector3d camera_offset):body_to_camera_(body_to_camera),camera_offset_(camera_offset)
{
    pxc = camera_offset(0);
    pyc = camera_offset(1);
    pzc = camera_offset(2);
}

std::vector<ttc_object_avoidance::FeatAndDepth> DepthPredictor::calculateDepth(Eigen::Vector3d velocity, Eigen::Quaterniond quat, Eigen::Vector3d angular_vel, std::vector<ttc_object_avoidance::FeatAndDisplacement> feats)
{
    Eigen::Matrix3d rot_mat(quat);
    Eigen::Vector3d velocity_camera = body_to_camera_ * rot_mat.transpose() * velocity;
    Eigen::Vector3d omega_camera = body_to_camera_ * angular_vel;
    double velx, vely, velz, wx, wy, wz;
    velx = velocity_camera(0);
    vely = velocity_camera(1);
    velz = velocity_camera(2);

    wx = omega_camera(0);
    wy = omega_camera(1);
    wz = omega_camera(2);

    std::vector<ttc_object_avoidance::FeatAndDepth> depthInfo;

    for (const auto& feat : feats) {
        double featx, featy, featvx, featvy;
        featx = feat.pt.x;
        featy = feat.pt.y;

        featvx = feat.displacement.x;
        featvy = feat.displacement.y;

        double pz_x = (pyc*wz - pzc*wy - velx -pxc*featx*wy + pyc*featx*wx +velz*featx)/(featvx - featy*wz + (1+pow(featx,2))*wy - featx*featy*wx);
        double pz_y =(-pxc*wz + pzc*wx - vely -pxc*featy*wy + pyc*featy*wx +velz*featy)/(featvy + featx*wz - (1+pow(featy,2))*wx + featx*featy*wy);

        double pz = (pz_x + pz_y)/2;
        double depth = pz*sqrt(1+pow(featx,2)+pow(featy,2));
        depth = (depth > 100) ? 100 : depth;

        ttc_object_avoidance::FeatAndDepth info;
        info.feat_id = feat.feat_id;
        info.pt = feat.pt;
        info.depth = depth;
        depthInfo.push_back(info);
    }

    return depthInfo;
}

}