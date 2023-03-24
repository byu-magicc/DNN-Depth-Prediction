#pragma once

#include "AbstractMap.h"
#include <memory>

#include <octomap/octomap.h>
#include <octomap/OcTree.h>
#include <octomap/Pointcloud.h>
#include <octomap/math/Quaternion.h>

namespace depth {

class OctoMap : public AbstractMap {
public:
    OctoMap();
    void insert(const std::vector<Eigen::Vector3d> &points, Eigen::Vector3d point_origin, Eigen::Quaterniond sensor_to_inertial);
    void saveMap(std::string filename);
private:
    std::unique_ptr<octomap::OcTree> tree;
    int id_counter = 0;
};

}