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

  Eigen::Vector3d t;
  t << 0.0, 0.0, 0.8;
  pinocchio::SE3 target(Eigen::Quaterniond::Identity(), t);

  CodeGenCost<double> cost_code_gen(model, 2, target);
  cost_code_gen.initLib();
  cost_code_gen.loadLib();

  auto q = pinocchio::neutral(model);
  // auto q = pinocchio::randomConfiguration(model);
  q.template segment<3>(0) << 0.0, 0.0, 1.0;
  VectorXd v = VectorXd::Zero(model.nv);
  v.template segment<3>(0) << 1.0, 0.0, 0.0;
  VectorXd u = VectorXd::Zero(dynamics_code_gen.nu);

  VectorXd x(dynamics_code_gen.nx);
  x << q, v;

  std::ofstream of("out.txt", std::ofstream::out | std::ofstream::trunc);

  for (size_t i = 0; i < 100; i++) {
    dynamics_code_gen.evalFunction(x, u);
    dynamics_code_gen.evalJacobian(x, u);

    cost_code_gen.evalFunction(x, u);
    cost_code_gen.evalJacobian(x, u);
    cost_code_gen.evalHessian(x, u);

    std::cout << i << ": \n" << cost_code_gen.f << std::endl;
    std::cout << cost_code_gen.df_dx.transpose() << '\n' << std::endl;
    std::cout << cost_code_gen.d2f_dx2 << '\n' << std::endl;
    std::cout << cost_code_gen.d2f_du2 << '\n' << std::endl;

    std::cout << dynamics_code_gen.f.transpose() << '\n' << std::endl;
    std::cout << dynamics_code_gen.df_dx << '\n' << std::endl;
    std::cout << dynamics_code_gen.df_du << '\n' << std::endl;

    of << dynamics_code_gen.f.transpose() << '\n';

    x = dynamics_code_gen.f;
  }
}