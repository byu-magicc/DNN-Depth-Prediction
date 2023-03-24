#pragma once

#include <vector>
#include <Eigen/Dense>

namespace depth {

class AbstractMap {
public:
    virtual void insert(const std::vector<Eigen::Vector3d> &points, Eigen::Vector3d point_origin, Eigen::Quaterniond sensor_to_inertial) = 0;
    virtual void saveMap(std::string filename) = 0;
};

}