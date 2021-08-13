#include <iostream>
#include <string>
#include <vector>

#define DYNAMICS_IMPL
#define DYNAMICS_CONTACT
#include "dynamics.hpp"

const char FILE_PATH[] =
    "/opt/openrobots/share/monoped_description/urdf/monoped.urdf";

int main() {
  pinocchio::Model model;
  pinocchio::JointModelFreeFlyer root_joint;
  pinocchio::urdf::buildModel(std::string(FILE_PATH), root_joint, model);
  pinocchio::Data data(model);

  CodeGenDynamics<double> dynamics_code_gen(model, 1.0, 0.01, 50, {10});
  dynamics_code_gen.initLib();
  dynamics_code_gen.loadLib();

  auto q = pinocchio::neutral(model);
  q(2) = 2;
  auto v = Eigen::VectorXd(model.nv);
  v.setZero();
  auto u = Eigen::VectorXd(dynamics_code_gen.nu);
  u.setZero();
  u(1) = 0.1;

  for (int i = 0; i < 100; ++i) {
    dynamics_code_gen.evalFunction(q, v, u);
    dynamics_code_gen.evalJacobian(q, v, u);
    q = dynamics_code_gen.qe;
    v = dynamics_code_gen.ve;
    std::cout << q.transpose() << std::endl;
  }
}