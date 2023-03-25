#include "depth/depth_predictor_node.h"
#include "depth/AbstractMap.h"
#include "depth/OctoMap.h"

#define OCTOMAP_DEPTH_MAP 0
#define BSPLINE_DETPH_MAP 1
#define DEPTH_MAP_TYPE OCTOMAP_DEPTH_MAP


namespace depth
{

DepthPredictorNode::DepthPredictorNode():
  nh_(), nh_private_("~"),
    #if DEPTH_MAP_TYPE == OCTOMAP_DEPTH_MAP
    map_(new OctoMap())
    #endif
{
    std::string imu_topic, feat_velocity_topic, odm_topic;
    if (!nh_private_.getParam("/imu_topic", imu_topic) || !nh_private_.getParam("/feat_velocity_topic", feat_velocity_topic)
        ||!nh_private_.getParam("/odometry_topic", odm_topic)) {
        ROS_FATAL("Need topic names!!");
    }

    nh_.subscribe(imu_topic, 1, &DepthPredictorNode::imu_callback, this);
    nh_.subscribe(odm_topic, 1, &DepthPredictorNode::odometry_callback, this);
    nh_.subscribe(feat_velocity_topic, 1, &DepthPredictorNode::feat_callback, this);

    // TODO: Publish feature depths?

    std::vector<double> body_to_cam_vec{0, 1, 0,
                                        0, 0, 1,
                                        1, 0, 0};
    nh_private_.param<std::vector<double>>("/body_to_camera_rotation", body_to_cam_vec, body_to_cam_vec);
    body_to_cam = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(body_to_cam_vec.data());
    std::vector<double> camera_offset_vec{0.0, 0.0, 0.3};
    nh_private_.param<std::vector<double>>("/camera_offset_from_body", camera_offset_vec, camera_offset_vec);
    camera_offset = Eigen::Map<Eigen::Matrix<double, 3, 1>>(camera_offset_vec.data());
    depth_predictor_ = std::unique_ptr<DepthPredictor>(new DepthPredictor(body_to_cam, camera_offset));
}

void DepthPredictorNode::feat_callback(const ttc_object_avoidance::TrackedFeatsWDis tracked_feats)
{
    auto points = depth_predictor_->calculateDepth(getAvgVelocity(), quat, getAvgOmega(), tracked_feats.feats);
    
    Eigen::Matrix3d rot_body_to_inertial(quat.inverse());
    
    Eigen::Vector3d camera_origin_intertial = position + Eigen::Matrix3d (quat).transpose() * camera_offset;
    Eigen::Quaterniond camera_to_inertial(body_to_cam.transpose()*rot_body_to_inertial);
    map_->insert(points,camera_origin_intertial, camera_to_inertial);
}

void DepthPredictorNode::odometry_callback(nav_msgs::OdometryConstPtr odm) {
    geometry_msgs::Vector3 vel = odm->twist.twist.linear;
    Eigen::Vector3d newVel({{vel.x, vel.y, vel.z}});
    velocity_total += newVel;
    num_velocity_measurements++;
    geometry_msgs::Point pos = odm->pose.pose.position;
    geometry_msgs::Quaternion quaternion = odm->pose.pose.orientation;

    position = Eigen::Vector3d({{pos.x, pos.y, pos.z}});
    quat = Eigen::Quaterniond(quaternion.w,quaternion.x, quaternion.y, quaternion.z);
}

void DepthPredictorNode::imu_callback(const sensor_msgs::ImuConstPtr imu_data) {
    Eigen::Vector3d newOmega({{imu_data->angular_velocity.x, imu_data->angular_velocity.y, imu_data->angular_velocity.z}});
    omega_total += newOmega;
    num_omega_measurements++;
}

Eigen::Vector3d DepthPredictorNode::getAvgOmega() {
    Eigen::Vector3d avg = omega_total / num_omega_measurements;
    omega_total = Eigen::Vector3d({{0, 0, 0}});
    num_omega_measurements = 0;
    return avg;
}

Eigen::Vector3d DepthPredictorNode::getAvgVelocity() {
    Eigen::Vector3d avg = velocity_total / num_velocity_measurements;
    velocity_total = Eigen::Vector3d({{0, 0, 0}});
    num_velocity_measurements = 0;
    return avg;
}

}



int main(int argc, char **argv) 
{
    ros::init(argc, argv, "depth_predictor");
    depth::DepthPredictorNode depth_tracker();

    ros::spin();

    return 0;
}