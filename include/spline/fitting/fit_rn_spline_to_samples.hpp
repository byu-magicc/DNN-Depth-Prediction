#pragma once

#include <vector>
#include <memory>

#include <ceres/ceres.h>
#include <Eigen/Dense>

#include "spline/rn_spline.hpp"
#include "utils/fit_rn_spline_residual.hpp"
#include "residuals/dummy_callback.h"

namespace utils
{

struct FitSplineParams
{
  int spline_k = 0;
  double spline_dt = 0.0;

  double start_time = 0.0;
  double end_time = 0.0;
};

template <int N>
class FitRnSplineToSamples
{
private:
  using Vector = Eigen::Matrix<double, N, 1>;
  using Matrix = Eigen::Matrix<double, N, N>;
  using Map = Eigen::Map<Vector>;
  using SplineT = spline::RnSpline<Map, N>;

public:
  FitRnSplineToSamples(FitSplineParams params) : spline_params_{params}, dummy_callback_{nullptr}
  {}

  std::shared_ptr<SplineT> fit(std::vector<std::pair<double, Vector>> const& samples)
  {
    // initialize control points
    std::shared_ptr<std::vector<Map>> control_points(new std::vector<Map>());
    int num_cp = static_cast<int>(std::ceil((spline_params_.end_time - spline_params_.start_time)/spline_params_.spline_dt)) + spline_params_.spline_k;
    for(int i = 0; i < num_cp; ++i) 
    {
      double* p = new double[N];
      for(int j = 0; j < N; ++j) p[j] = 0.0;
      control_points->emplace_back(p); // TODO: better initialization
    }

    // create spline
    std::shared_ptr<SplineT> spline(new SplineT(spline_params_.spline_k));
    spline->init_ctrl_pts(control_points, spline_params_.start_time, spline_params_.spline_dt);

    // make problem

    // add dummy evaluation callback. Memory provided using AddParameterBlock only
    // gets overwritten at end of optimization, unless problem has an evaluation
    // callback, in which case the memory is overwritten every time the residuals
    // are evaluated.
    ceres::Problem::Options problem_options;
    if (dummy_callback_ == nullptr) dummy_callback_.reset(new DummyCallback());
    problem_options.evaluation_callback = dummy_callback_.get();
    ceres::Problem problem(problem_options);

    for(auto const& sample : samples)
    {
      std::vector<double*> params;
      int i = spline->get_i(sample.first);
      for(int j = 0; j < spline->get_order(); ++j) params.push_back(control_points->at(i + 1 + j).data());
      ceres::CostFunction* resid = new FitRnSplineResidual<2>(sample.second, sample.first, spline);
      problem.AddResidualBlock(resid, nullptr, params);
    }

    // solve
    ceres::Solver::Options solver_options;
    solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    solver_options.minimizer_progress_to_stdout = true;
    solver_options.max_num_iterations = 50;

    ceres::Solver::Summary summary;
    ceres::Solve(solver_options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    return spline;
  }

private:
  FitSplineParams spline_params_;

  std::shared_ptr<DummyCallback> dummy_callback_;

};

} // namespace utils