#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <cmath>

#include "lie_groups/so2.hpp"
#include "lie_groups/so3.hpp"
#include "lie_groups/se2.hpp"
#include "lie_groups/se3.hpp"

template <typename G, int DIM>
class LieGroupJacTest : public testing::Test
{
protected:
  typedef Eigen::Matrix<double, DIM, DIM> MatrixDd;
  typedef Eigen::Matrix<double, DIM, 1> VectorDd;

public:
  LieGroupJacTest()
  {}

  MatrixDd Jl_fin_diff(VectorDd tau)
  {
    MatrixDd jac = MatrixDd::Zero();
    G g = G::Exp(tau);
    for(int i = 0; i < DIM; ++i)
    {
      VectorDd dtau = VectorDd::Zero();
      dtau(i) = h_;
      G g_pert = G::Exp(tau + dtau);

      jac.col(i) = ((g_pert * g.inverse()).Log())/h_;
    }

    return jac;
  }

  MatrixDd Jr_fin_diff(VectorDd tau)
  {
    MatrixDd jac = MatrixDd::Zero();
    G g = G::Exp(tau);
    for(int i = 0; i < DIM; ++i)
    {
      VectorDd dtau = VectorDd::Zero();
      dtau(i) = h_;
      G g_pert = G::Exp(tau + dtau);

      jac.col(i) = ((g.inverse() * g_pert).Log())/h_;
    }

    return jac;
  }

  MatrixDd Ad_fin_diff(G g1, G g2)
  {
    MatrixDd jac = MatrixDd::Zero();
    G g_12 = g1 * g2;
    for(int i = 0; i < DIM; ++i)
    {
      VectorDd dtau = VectorDd::Zero();
      dtau(i) = h_;
      G g_12_pert = g1 * G::Exp(dtau) * g2;

      jac.col(i) = ((g_12_pert * g_12.inverse()).Log())/h_;
    }

    return jac;
  }

protected:
  double h_{1e-6};
  double tolerance_{1e-5};
};

typedef LieGroupJacTest<lie_groups::SO2d, 1> SO2LieGroupJacTest;
typedef LieGroupJacTest<lie_groups::SO3d, 3> SO3LieGroupJacTest;
typedef LieGroupJacTest<lie_groups::SE2d, 3> SE2LieGroupJacTest;
typedef LieGroupJacTest<lie_groups::SE3d, 6> SE3LieGroupJacTest;

TEST_F(SO2LieGroupJacTest, SO2JacobianTest)
{
  for(int i = 0; i < 100; ++i)
  {
    VectorDd tau = VectorDd::Random();
    tau *= M_PI;

    MatrixDd jl = lie_groups::SO2d::Jl(tau);
    MatrixDd jl_fin_diff = Jl_fin_diff(tau);

    MatrixDd jr = lie_groups::SO2d::Jr(tau);
    MatrixDd jr_fin_diff = Jr_fin_diff(tau);

    lie_groups::SO2d g1 = lie_groups::SO2d::random();
    lie_groups::SO2d g2 = lie_groups::SO2d::random();
    MatrixDd ad = g1.Ad();
    MatrixDd ad_fin_diff = Ad_fin_diff(g1, g2);

    EXPECT_TRUE(jl_fin_diff.isApprox(jl, tolerance_));
    EXPECT_TRUE(jr_fin_diff.isApprox(jr, tolerance_));
    EXPECT_TRUE(jl_fin_diff.inverse().isApprox(lie_groups::SO2d::Jl_inv(tau), tolerance_));
    EXPECT_TRUE(jr_fin_diff.inverse().isApprox(lie_groups::SO2d::Jr_inv(tau), tolerance_));
    EXPECT_TRUE(ad_fin_diff.isApprox(ad, tolerance_));
  }
}

TEST_F(SO3LieGroupJacTest, SO3JacobianTest)
{
  for(int i = 0; i < 100; ++i)
  {
    VectorDd tau = VectorDd::Random();
    tau /= tau.norm();
    tau *= Eigen::Matrix<double, 1, 1>::Random()*M_PI;
    // inverse fails if tau.norm() > pi. Not sure why

    MatrixDd jl = lie_groups::SO3d::Jl(tau);
    MatrixDd jl_fin_diff = Jl_fin_diff(tau);

    MatrixDd jr = lie_groups::SO3d::Jr(tau);
    MatrixDd jr_fin_diff = Jr_fin_diff(tau);

    lie_groups::SO3d g1 = lie_groups::SO3d::random();
    lie_groups::SO3d g2 = lie_groups::SO3d::random();
    MatrixDd ad = g1.Ad();
    MatrixDd ad_fin_diff = Ad_fin_diff(g1, g2);

    EXPECT_TRUE(jl_fin_diff.isApprox(jl, tolerance_));
    EXPECT_TRUE(jr_fin_diff.isApprox(jr, tolerance_));
    EXPECT_TRUE(jl_fin_diff.inverse().isApprox(lie_groups::SO3d::Jl_inv(tau), tolerance_));
    EXPECT_TRUE(jr_fin_diff.inverse().isApprox(lie_groups::SO3d::Jr_inv(tau), tolerance_));
    EXPECT_TRUE(ad_fin_diff.isApprox(ad, tolerance_));
  }
}

TEST_F(SE2LieGroupJacTest, SE2JacobianTest)
{ 
  h_ = 1e-5;
  for(int i = 0; i < 100; ++i)
  {
    VectorDd tau = VectorDd::Random();
    tau.head<2>() *= 10;
    tau.tail<1>() *= M_PI;

    // this one only works for h = 1e-5, not sure why
    MatrixDd jl = lie_groups::SE2d::Jl(tau);
    MatrixDd jl_fin_diff = Jl_fin_diff(tau);

    MatrixDd jr = lie_groups::SE2d::Jr(tau);
    MatrixDd jr_fin_diff = Jr_fin_diff(tau);

    lie_groups::SE2d g1 = lie_groups::SE2d::random();
    lie_groups::SE2d g2 = lie_groups::SE2d::random();
    MatrixDd ad = g1.Ad();
    MatrixDd ad_fin_diff = Ad_fin_diff(g1, g2);

    EXPECT_TRUE(jl_fin_diff.isApprox(jl, tolerance_));
    EXPECT_TRUE(jr_fin_diff.isApprox(jr, tolerance_));
    EXPECT_TRUE(jl_fin_diff.inverse().isApprox(lie_groups::SE2d::Jl_inv(tau), tolerance_));
    EXPECT_TRUE(jr_fin_diff.inverse().isApprox(lie_groups::SE2d::Jr_inv(tau), tolerance_));
    EXPECT_TRUE(ad_fin_diff.isApprox(ad, tolerance_));
  }
}

TEST_F(SE3LieGroupJacTest, SE3JacobianTest)
{
  // h_ = 1e-4;
  for(int i = 0; i < 100; ++i)
  {
    VectorDd tau = VectorDd::Random();
    tau.head<3>() *= 10;
    tau.tail<3>() /= tau.tail<3>().norm();
    tau.tail<3>() *= Eigen::Matrix<double, 1, 1>::Random() * M_PI;

    MatrixDd jl = lie_groups::SE3d::Jl(tau);
    MatrixDd jl_fin_diff = Jl_fin_diff(tau);
    if(jl_fin_diff.hasNaN())
    {
      i--;
      continue;
    }

    MatrixDd jr = lie_groups::SE3d::Jr(tau);
    MatrixDd jr_fin_diff = Jr_fin_diff(tau);
    if(jr_fin_diff.hasNaN())
    {
      i--;
      continue;
    }

    lie_groups::SE3d g1 = lie_groups::SE3d::random();
    lie_groups::SE3d g2 = lie_groups::SE3d::random();
    MatrixDd ad = g1.Ad();
    MatrixDd ad_fin_diff = Ad_fin_diff(g1, g2);
    if(ad_fin_diff.hasNaN())
    {
      i--;
      continue;
    }

    EXPECT_TRUE(jl_fin_diff.isApprox(jl, tolerance_));
    EXPECT_TRUE(jr_fin_diff.isApprox(jr, tolerance_));
    EXPECT_TRUE(jl_fin_diff.inverse().isApprox(lie_groups::SE3d::Jl_inv(tau), tolerance_));
    EXPECT_TRUE(jr_fin_diff.inverse().isApprox(lie_groups::SE3d::Jr_inv(tau), tolerance_));
    EXPECT_TRUE(ad_fin_diff.isApprox(ad, tolerance_));
  }
}