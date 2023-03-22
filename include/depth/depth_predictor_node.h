#pragma once

#include <vector>
#include <ros/ros.h>
#include <Eigen/Dense>
#include <memory>
#include <sensor_msgs/Imu.h>

#include "depth/depth_predictor.h"
#include "ttc_object_avoidance/TrackedFeatsWDis.h"

namespace depth
{

class DepthPredictorNode
{
public:
    DepthPredictorNode();

private:
    void feat_callback(const ttc_object_avoidance::TrackedFeatsWDis feats_w_displacement);
    void velocity_callback(const geometry_msgs::Vector3ConstPtr vel);
    void imu_callback(const sensor_msgs::ImuConstPtr msg);

    Eigen::Vector3d getAvgOmega();
    Eigen::Vector3d getAvgVelocity();


    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;
    ros::Publisher depth_pub_;

    std::unique_ptr<DepthPredictor> depth_predictor_;
    Eigen::Vector3d velocity_total;
    int num_velocity_measurements = 0;
    Eigen::Quaterniond quat;
    Eigen::Vector3d omega_total;
    int num_omega_measurements = 0;
};


}