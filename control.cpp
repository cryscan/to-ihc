#include <iostream>
#include <vector>

#include <pinocchio/codegen/cppadcg.hpp>
#include <pinocchio/parsers/urdf.hpp>

#define DYNAMICS_IMPL
#include "cost.hpp"
#include "dynamics.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

const char FILE_PATH[] =
    "/opt/openrobots/share/monoped_description/urdf/monoped.urdf";

int main() {
  pinocchio::Model model;
  pinocchio::JointModelFreeFlyer root_joint;
  pinocchio::urdf::buildModel(std::string(FILE_PATH), root_joint, model);
  pinocchio::Data data(model);

  double dt = 0.01;
  double mu = 1.0;

  CodeGenDynamics<double> dynamics_code_gen(model, dt, 50, mu, {10});
  dynamics_code_gen.initLib();
  dynamics_code_gen.loadLib();

  CodeGenCost<double> cost_code_gen(model);
  cost_code_gen.initLib();
  cost_code_gen.loadLib();

  auto q = pinocchio::neutral(model);
  // auto q = pinocchio::randomConfiguration(model);
  q.template segment<3>(0) << 0.0, 0.0, 1.0;
  auto v = VectorXd(model.nv);
  v.setZero();
  auto u = VectorXd(dynamics_code_gen.nu);
  u.setZero();

  VectorXd x(dynamics_code_gen.nx);
  x << q, v;

  VectorXd t(dynamics_code_gen.nx);
  t.setZero();
  t.template segment<3>(0) << 0.0, 0.0, 0.8;
}