#ifndef __COST_HPP__
#define __COST_HPP__

#include <pinocchio/codegen/code-generator-base.hpp>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

template <typename Scalar>
struct CodeGenCost : public pinocchio::CodeGenBase<Scalar> {
  typedef pinocchio::FrameIndex FrameIndex;
  typedef pinocchio::CodeGenBase<Scalar> Base;
  typedef typename Base::Model Model;
  typedef typename Base::ADConfigVectorType ADConfigVectorType;
  typedef typename Base::ADTangentVectorType ADTangentVectorType;
  typedef typename Base::MatrixXs MatrixXs;
  typedef typename Base::VectorXs VectorXs;
  typedef typename Base::ADScalar ADScalar;
  typedef typename Base::ADMatrixXs ADMatrixXs;
  typedef typename Base::ADVectorXs ADVectorXs;

  CodeGenCost(const Model &model, const std::string &function_name = "cost",
              const std::string &library_name = "cg_cost_eval")
      : Base(model, model.nq + 2 * model.nv - 6, 1, function_name,
             library_name),
        nx(model.nq + model.nv), nu(model.nv - 6) {
    ad_q = ADConfigVectorType(model.nq);
    ad_q = pinocchio::neutral(model);
    ad_v = ADTangentVectorType(model.nv);
    ad_v.setZero();
    ad_u = ADTangentVectorType(nu);
    ad_u.setZero();

    x = VectorXs::Zero(Base::getInputDimension());
    res = VectorXs::Zero(Base::getOutputDimension());

    df_dx = MatrixXs::Zero(nx, nx);
    df_du = MatrixXs::Zero(nx, nu);
  }

  void buildMap() {
    CppAD::Independent(ad_X);

    ad_fun.Dependent(ad_X, ad_Y);
    ad_fun.optimize("no_compare_op");
  }

  void initLib() {
    Base::initLib();
    Base::cgen_ptr->setCreateHessian(true);
  }

  VectorXs g;
  MatrixXs df_dx, df_du;

  const int nx;
  const int nu;

protected:
  using Base::ad_data;
  using Base::ad_fun;
  using Base::ad_model;
  using Base::ad_q;
  using Base::ad_v;
  using Base::ad_X;
  using Base::ad_Y;
  using Base::jac;
  using Base::y;

  ADTangentVectorType ad_u;

  VectorXs x;
  VectorXs res;
};

#endif