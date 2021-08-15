#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

#include <pinocchio/codegen/cppadcg.hpp>
#include <pinocchio/parsers/urdf.hpp>

#define DYNAMICS_IMPL
#include "cost.hpp"
#include "dynamics.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

template <typename Scalar> struct LQR {
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXs;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;

  template <typename Vector>
  LQR(CodeGenDynamics<Scalar> &dynamics, CodeGenCost<Scalar> &cost,
      size_t horizon, const Eigen::MatrixBase<Vector> &x0)
      : dynamics(dynamics), cost(cost), horizon(horizon),
        x(horizon + 1, VectorXs(dynamics.nx)),
        u(horizon, VectorXs(dynamics.nu)),
        A(horizon, MatrixXs::Identity(dynamics.nx + 1, dynamics.nx + 1)),
        B(horizon, MatrixXs::Zero(dynamics.nx + 1, dynamics.nu)),
        Q(horizon, MatrixXs(dynamics.nx + 1, dynamics.nx + 1)),
        R(horizon, MatrixXs(dynamics.nu, dynamics.nu)),
        K(horizon + 1, MatrixXs::Zero(dynamics.nu, dynamics.nx + 1)),
        P(horizon + 1, MatrixXs::Zero(dynamics.nx + 1, dynamics.nx + 1)) {
    assert(dynamics.nx == cost.nx);
    assert(dynamics.nu == cost.nu);

    x[0] = x0;
  }

  /*
  // populate x, linearized dynamics and quadratized cost
  void rollout() {
    for (size_t i = 0; i < horizon; i++) {
      dynamics.evalFunction(x[i], u[i]);
      dynamics.evalJacobian(x[i], u[i]);

      x[i + 1] = dynamics.f;

      auto nx = dynamics.nx;
      A[i].block(0, 0, nx, nx) << dynamics.df_dx;
      B[i].topRows(nx) << dynamics.df_du;

      cost.evalFunction(x[i], u[i]);
      cost.evalJacobian(x[i], u[i]);
      cost.evalHessian(x[i], u[i]);

      Q[i] << cost.d2f_dx2, cost.df_dx, cost.df_dx.transpose(), 2 * cost.f;
      Q[i] *= 0.5;
      R[i] << 0.5 * cost.d2f_du2;

      auto remove_nan = [](auto x) { return std::isfinite(x) ? x : Scalar(0); };
      A[i] = A[i].unaryExpr(remove_nan);
      B[i] = B[i].unaryExpr(remove_nan);
      Q[i] = Q[i].unaryExpr(remove_nan);
      R[i] = R[i].unaryExpr(remove_nan);
    }
  }
  */

  void solve() {
    for (size_t i = 1; i <= horizon; i++) {
      const auto &a = A[horizon - i];
      const auto &b = B[horizon - i];
      const auto &q = Q[horizon - i];
      const auto &r = R[horizon - i];

      const auto &p = P[i - 1];

      auto remove_nan = [](auto x) { return std::isfinite(x) ? x : Scalar(0); };

      auto &k = K[i];
      k = -(r + b.transpose() * p * b).ldlt().solve(b.transpose() * p * a);
      k = k.unaryExpr(remove_nan);

      auto a_bk = a + b * k;
      P[i] = q + k.transpose() * r * k + a_bk.transpose() * p * a_bk;
      P[i] = P[i].unaryExpr(remove_nan);
    }
  }

  VectorXs policy(const VectorXs &dx, const MatrixXs &K) {
    VectorXs z(dx.size() + 1);
    z << dx, Scalar(1);
    return K * z;
  }

  void update() {
    std::vector<VectorXs> x_(x);
    std::vector<VectorXs> u_(u);

    for (size_t i = 0; i < horizon; i++) {
      auto dx = x_[i] - x[i];
      auto du = policy(dx, K[horizon - i]);

      u_[i] += du;

      dynamics.evalFunction(x_[i], u_[i]);
      dynamics.evalJacobian(x_[i], u_[i]);

      x_[i + 1] = dynamics.f;

      auto nx = dynamics.nx;
      A[i].block(0, 0, nx, nx) << dynamics.df_dx;
      B[i].topRows(nx) << dynamics.df_du;

      cost.evalFunction(x_[i], u_[i], x[i], u[i]);
      cost.evalJacobian(x_[i], u_[i], x[i], u[i]);
      cost.evalHessian(x_[i], u_[i], x[i], u[i]);

      Q[i] << cost.d2f_dx2, cost.df_dx, cost.df_dx.transpose(), 2 * cost.f;
      Q[i] *= 0.5;
      R[i] << 0.5 * cost.d2f_du2;
    }

    std::swap(x_, x);
    std::swap(u_, u);
  }

  void print(std::ostream &os, bool verbose = true) {
    using std::endl;

    if (verbose) {
      for (size_t i = 0; i < horizon; i++) {
        os << i << ": \n";
        os << "x: " << x[i].transpose() << '\n'
           << "u: " << u[i].transpose() << '\n'
           << endl;
        os << "A:\n" << A[i] << '\n' << endl;
        os << "B:\n" << B[i] << '\n' << endl;
        os << "Q:\n" << Q[i] << '\n' << endl;
        os << "R:\n" << R[i] << '\n' << endl;
      }

      os << horizon << ": \n";
      os << "x: " << x[horizon].transpose() << endl;
    } else {
      for (size_t i = 0; i < horizon; i++) {
        os << x[i].transpose() << '\t' << u[i].transpose() << endl;
      }
    }
  }

  const size_t horizon;

private:
  std::vector<VectorXs> x, u;
  std::vector<MatrixXs> A, B, Q, R;
  std::vector<MatrixXs> K, P;

  CodeGenDynamics<Scalar> &dynamics;
  CodeGenCost<Scalar> &cost;
};

const char FILE_PATH[] =
    "/opt/openrobots/share/monoped_description/urdf/monoped.urdf";

int main() {
  std::ifstream fs("in.txt");
  std::ofstream of("out.txt", std::ofstream::out | std::ofstream::trunc);

  pinocchio::Model model;
  pinocchio::JointModelFreeFlyer root_joint;
  pinocchio::urdf::buildModel(std::string(FILE_PATH), root_joint, model);
  pinocchio::Data data(model);

  double dt = 0.01;
  double mu = 1.0;

  CodeGenDynamics<double> dynamics_code_gen(model, dt, 100, mu, 1.0, {10});
  dynamics_code_gen.initLib();
  dynamics_code_gen.loadLib();

  size_t horizon, iters;
  fs >> horizon >> iters;

  Eigen::Vector3d t;
  {
    double tx, ty, tz;
    fs >> tx >> ty >> tz;
    t << tx, ty, tz;
  }
  pinocchio::SE3 target(Eigen::Quaterniond::Identity(), t);

  CodeGenCost<double> cost_code_gen(model, 2, target, 0.5);
  cost_code_gen.initLib();
  cost_code_gen.loadLib();

  auto q = pinocchio::neutral(model);
  // auto q = pinocchio::randomConfiguration(model);
  q.template segment<3>(0) << 0.0, 0.0, 1.0;
  VectorXd v = VectorXd::Zero(model.nv);
  VectorXd u = VectorXd::Zero(dynamics_code_gen.nu);

  {
    double vx, vy, vz;
    fs >> vx >> vy >> vz;
    v.template segment<3>(0) << vx, vy, vz;
  }

  VectorXd x(dynamics_code_gen.nx);
  x << q, v;

  LQR<double> lqr(dynamics_code_gen, cost_code_gen, horizon, x);

  for (size_t i = 0; i < iters; i++) {
    lqr.update();
    lqr.solve();
  }

  lqr.print(std::cout);
  lqr.print(of, false);
}