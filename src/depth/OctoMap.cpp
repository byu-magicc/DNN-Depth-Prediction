#include "depth/OctoMap.h"

namespace depth {

OctoMap::OctoMap() : tree(new octomap::OcTree(0.1)) {
    id_counter = 0;
}

void OctoMap::insert(const std::vector<Eigen::Vector3d>& points, Eigen::Vector3d point_origin, Eigen::Quaterniond sensor_to_inertial) {
    octomap::Pointcloud* cloud = new octomap::Pointcloud();

    for (const auto &point: points) {
        cloud->push_back(octomap::point3d(point(0), point(1), point(2)));
    }
    octomap::pose6d pose(octomap::point3d(point_origin(0), point_origin(1), point_origin(2)),
                        octomath::Quaternion(sensor_to_inertial.w(), sensor_to_inertial.x(), sensor_to_inertial.y(), sensor_to_inertial.z()));
    octomap::ScanNode node(cloud, pose, id_counter++);
    tree->insertPointCloud(node,99);
}

void OctoMap::saveMap(std::string filename) {
    tree->writeBinary(filename);
}


}

