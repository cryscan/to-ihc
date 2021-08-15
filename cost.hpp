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

  CodeGenCost(const Model &model, FrameIndex base_idx,
              const pinocchio::SE3 &target, Scalar alpha = 0.9,
              const std::string &function_name = "cost",
              const std::string &library_name = "cg_cost_eval")
      : Base(model, 2 * (model.nq + 2 * model.nv - 6), 1, function_name,
             library_name),
        nx(model.nq + model.nv), nu(model.nv - 6), base_idx(base_idx),
        target(target.template cast<ADScalar>()), alpha(alpha) {
    ad_q = ADConfigVectorType(model.nq);
    ad_q = pinocchio::neutral(model);
    ad_v = ADTangentVectorType(model.nv);
    ad_v.setZero();

    ad_u = ADVectorXs::Zero(nu);
    ad_ox = ADVectorXs::Zero(nx);
    ad_ou = ADVectorXs::Zero(nu);

    hess = MatrixXs::Zero(ad_X.size(), ad_X.size());

    x = VectorXs::Zero(Base::getInputDimension());
    res = VectorXs::Zero(Base::getOutputDimension());

    df_dx = VectorXs::Zero(nx);
    df_du = VectorXs::Zero(nu);

    d2f_dx2 = MatrixXs::Zero(nx, nx);
    d2f_du2 = MatrixXs::Zero(nu, nu);
  }

  void buildMap() {
    CppAD::Independent(ad_X);

    Eigen::DenseIndex it = 0;
    ad_q = ad_X.segment(it, ad_model.nq);
    it += ad_model.nq;
    ad_v = ad_X.segment(it, ad_model.nv);
    it += ad_model.nv;
    ad_u = ad_X.segment(it, nu);
    it += nu;

    ad_ox = ad_X.segment(it, nx);
    it += nx;
    ad_ou = ad_X.segment(it, nu);
    it += nu;

    cost();
    ad_Y(0) = ad_f;

    ad_fun.Dependent(ad_X, ad_Y);
    ad_fun.optimize("no_compare_op");
  }

  void initLib() {
    Base::initLib();
    Base::cgen_ptr->setCreateHessian(true);
  }

  using Base::evalFunction;
  template <typename StateVector, typename ActionVector>
  void evalFunction(const Eigen::MatrixBase<StateVector> &s,
                    const Eigen::MatrixBase<ActionVector> &u,
                    const Eigen::MatrixBase<StateVector> &os,
                    const Eigen::MatrixBase<ActionVector> &ou) {
    // fill x
    Eigen::DenseIndex it = 0;
    x.segment(it, nx) = s;
    it += nx;
    x.segment(it, nu) = u;
    it += nu;
    x.segment(it, nx) = os;
    it += nx;
    x.segment(it, nu) = ou;
    it += nu;

    evalFunction(x);
    res = Base::y;
    f = res(0);
  }

  using Base::evalJacobian;
  template <typename StateVector, typename ActionVector>
  void evalJacobian(const Eigen::MatrixBase<StateVector> &s,
                    const Eigen::MatrixBase<ActionVector> &u,
                    const Eigen::MatrixBase<StateVector> &os,
                    const Eigen::MatrixBase<ActionVector> &ou) {
    // fill x
    Eigen::DenseIndex it = 0;
    x.segment(it, nx) = s;
    it += nx;
    x.segment(it, nu) = u;
    it += nu;
    x.segment(it, nx) = os;
    it += nx;
    x.segment(it, nu) = ou;
    it += nu;

    evalJacobian(x);
    it = 0;
    df_dx = Base::jac.middleCols(it, nx).transpose();
    it += nx;
    df_du = Base::jac.middleCols(it, nu).transpose();
    it += nu;
  }

  template <typename Vector>
  void evalHessian(const Eigen::MatrixBase<Vector> &x,
                   const Eigen::MatrixBase<Vector> &w) {
    CppAD::cg::ArrayView<const Scalar> x_(
        PINOCCHIO_EIGEN_CONST_CAST(Vector, x).data(), (size_t)x.size());
    CppAD::cg::ArrayView<const Scalar> w_(
        PINOCCHIO_EIGEN_CONST_CAST(Vector, w).data(), (size_t)w.size());
    CppAD::cg::ArrayView<Scalar> hess_(hess.data(), (size_t)hess.size());

    // auto hess_ = Base::generatedFun_ptr->Hessian(x, 0);
    Base::generatedFun_ptr->Hessian(x_, w_, hess_);
  }

  template <typename StateVector, typename ActionVector>
  void evalHessian(const Eigen::MatrixBase<StateVector> &s,
                   const Eigen::MatrixBase<ActionVector> &u,
                   const Eigen::MatrixBase<StateVector> &os,
                   const Eigen::MatrixBase<ActionVector> &ou) {
    // fill x
    Eigen::DenseIndex it = 0;
    x.segment(it, nx) = s;
    it += nx;
    x.segment(it, nu) = u;
    it += nu;
    x.segment(it, nx) = os;
    it += nx;
    x.segment(it, nu) = ou;
    it += nu;

    VectorXs w(1);
    w << Scalar(1.0);

    evalHessian(x, w);
    it = 0;
    d2f_dx2 = hess.block(it, it, nx, nx);
    it += nx;
    d2f_du2 = hess.block(it, it, nu, nu);
    it += nu;
  }

  Scalar f;
  VectorXs df_dx, df_du;
  MatrixXs d2f_dx2, d2f_du2;

  const int nx;
  const int nu;

  const Scalar alpha;

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

  ADScalar ad_f;
  ADVectorXs ad_u, ad_ox, ad_ou;

  MatrixXs hess;

  VectorXs x;
  VectorXs res;

  const FrameIndex base_idx;
  const pinocchio::SE3Tpl<ADScalar> target;

  inline void cost();
};

template <typename Scalar> void CodeGenCost<Scalar>::cost() {
  pinocchio::framesForwardKinematics(ad_model, ad_data, ad_q);

  auto base_trans = ad_data.oMf[base_idx].translation();
  auto target_trans = target.translation();
  ADScalar ad_g = (base_trans - target_trans).squaredNorm();

  // ADScalar ang_dist = base_rot.angularDistance(target_rot);
  // ad_g += ang_dist * ang_dist;

  ad_g += ad_u.squaredNorm();

  ADVectorXs ad_x(nx);
  ad_x << ad_q, ad_v;

  ADScalar ad_h = (ad_x - ad_ox).squaredNorm() + (ad_u - ad_ou).squaredNorm();

  ad_f = (1 - alpha) * ad_g + alpha * ad_h;
}

#endif