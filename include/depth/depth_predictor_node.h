#pragma once

#include <vector>
#include <sstream>
#include <ros/ros.h>
#include <Eigen/Dense>
#include <memory>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>

#include "depth/depth_predictor.h"
#include "ttc_object_avoidance/TrackedFeatsWDis.h"
#include "ttc_object_avoidance/SaveMap.h"
#include "AbstractMap.h"
#include "OctoMap.h"

namespace depth
{

class DepthPredictorNode
{
public:
    DepthPredictorNode();

private:
    void feat_callback(const ttc_object_avoidance::TrackedFeatsWDis feats_w_displacement);
    void odometry_callback(const nav_msgs::OdometryConstPtr odm);
    void imu_callback(const sensor_msgs::ImuConstPtr msg);

    bool save_map_func(ttc_object_avoidance::SaveMap::Request &req, ttc_object_avoidance::SaveMap::Response &res);

    Eigen::Vector3d getAvgOmega();
    Eigen::Vector3d getAvgVelocity();


    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;

    ros::Subscriber imu_sub_;
    ros::Subscriber state_sub_;
    ros::Subscriber feat_sub_;
    ros::Publisher depth_pub_;
    ros::ServiceServer server_;

    std::unique_ptr<DepthPredictor> depth_predictor_;
    std::unique_ptr<AbstractMap> map_;

    Eigen::Vector3d camera_offset;
    Eigen::Matrix3d body_to_cam;

    Eigen::Vector3d velocity_total;
    int num_velocity_measurements = 0;
    Eigen::Quaterniond quat;
    Eigen::Vector3d omega_total;
    Eigen::Vector3d position;
    int num_omega_measurements = 0;
};


}