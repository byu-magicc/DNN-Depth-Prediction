#include "depth/depth_predictor.h"

namespace depth
{

DepthPredictor::DepthPredictor(Eigen::Matrix3d body_to_camera, Eigen::Vector3d camera_offset):body_to_camera_(body_to_camera),camera_offset_(camera_offset)
{
    Eigen::Vector3d body_offset = body_to_camera.transpose() * camera_offset;

    pxc = body_offset(0);
    pyc = body_offset(1);
    pzc = body_offset(2);
}

std::vector<Eigen::Vector3d> DepthPredictor::calculateDepth(Eigen::Vector3d body_velocity, Eigen::Quaterniond body_quat, Eigen::Vector3d body_angular_vel, std::vector<ttc_object_avoidance::FeatAndDisplacement> camera_feats)
{
    Eigen::Matrix3d rot_mat(body_quat);
    Eigen::Vector3d velocity_camera = body_to_camera_ * rot_mat.transpose() * body_velocity;
    Eigen::Vector3d omega_camera = body_to_camera_ * body_angular_vel;
    double velx, vely, velz, wx, wy, wz;
    velx = velocity_camera(0);
    vely = velocity_camera(1);
    velz = velocity_camera(2);

    wx = omega_camera(0);
    wy = omega_camera(1);
    wz = omega_camera(2);

    std::vector<Eigen::Vector3d> points;

    for (const auto& feat : camera_feats) {

        

        double featx, featy, featvx, featvy;
        featx = feat.pt.x;
        featy = feat.pt.y;

        featvx = feat.displacement.x;
        featvy = feat.displacement.y;

        double pz_x = (pyc*wz - pzc*wy - velx -pxc*featx*wy + pyc*featx*wx +velz*featx)/(featvx - featy*wz + (1+pow(featx,2))*wy - featx*featy*wx);
        double pz_y =(-pxc*wz + pzc*wx - vely -pxc*featy*wy + pyc*featy*wx +velz*featy)/(featvy + featx*wz - (1+pow(featy,2))*wx + featx*featy*wy);

        double pz = (pz_x + pz_y)/2;
        
        points.push_back(Eigen::Vector3d ({{featx*pz, featy*pz, pz}}));
    }

    return points;
}

}