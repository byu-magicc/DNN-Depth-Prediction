#pragma once

#include <vector>
#include <memory>

#include <ceres/ceres.h>
#include <Eigen/Dense>

#include <spline/rn_spline.hpp>

namespace utils
{

template <int N>
class FitRnSplineResidual : public ceres::CostFunction
{
private:
  using Vector = Eigen::Matrix<double, N, 1>;
  using Matrix = Eigen::Matrix<double, N, N>;
  using SplineT = spline::RnSpline<Eigen::Map<Vector>, N>;

public:
  FitRnSplineResidual(Vector sample, double t, std::shared_ptr<SplineT> spl) :
    sample_{sample}, t_{t}, spline_{spl}
  {
    set_num_residuals(N);
    std::vector<int32_t> param_block_sizes;
    for(int i = 0; i < spline_->get_order(); ++i) param_block_sizes.push_back(N);
    *mutable_parameter_block_sizes() = param_block_sizes;
  }

  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
  {
    Vector sample_spline;
    std::vector<Matrix> jacs;
    spline_->eval(t_, &sample_spline, nullptr, nullptr, (jacobians == nullptr) ? nullptr : &jacs);

    Eigen::Map<Vector> r(residuals);
    r = sample_spline - sample_;

    if(jacobians != nullptr)
    {
      for(int i = 0; i < spline_->get_order(); ++i)
      {
        if(jacobians[i] != nullptr)
        {
          Eigen::Map<Eigen::Matrix<double, N, N, Eigen::RowMajor>> J(jacobians[i]);
          J = jacs[i];
        }
      }
    }

    return true;
  }

private:
  std::shared_ptr<SplineT> spline_;

  Vector sample_;
  double t_;
};

} // namespace utils