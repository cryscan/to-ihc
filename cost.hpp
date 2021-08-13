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
              const pinocchio::SE3 &target,
              const std::string &function_name = "cost",
              const std::string &library_name = "cg_cost_eval")
      : Base(model, model.nq + 2 * model.nv - 6, 1, function_name,
             library_name),
        nx(model.nq + model.nv), nu(model.nv - 6), base_idx(base_idx),
        target(target.template cast<ADScalar>()) {
    ad_q = ADConfigVectorType(model.nq);
    ad_q = pinocchio::neutral(model);
    ad_v = ADTangentVectorType(model.nv);
    ad_v.setZero();
    ad_u = ADTangentVectorType(nu);
    ad_u.setZero();

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
  template <typename StateVector, typename TangentVector>
  void evalFunction(const Eigen::MatrixBase<StateVector> &s,
                    const Eigen::MatrixBase<TangentVector> &u) {
    // fill x
    Eigen::DenseIndex it = 0;
    x.segment(it, nx) = s;
    it += nx;
    x.segment(it, nu) = u;
    it += nu;

    evalFunction(x);
    res = Base::y;
    f = res(0);
  }

  using Base::evalJacobian;
  template <typename StateVector, typename TangentVector>
  void evalJacobian(const Eigen::MatrixBase<StateVector> &s,
                    const Eigen::MatrixBase<TangentVector> &u) {
    // fill x
    Eigen::DenseIndex it = 0;
    x.segment(it, nx) = s;
    it += nx;
    x.segment(it, nu) = u;
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
                   const Eigen::MatrixBase<Vector> &w, int) {
    CppAD::cg::ArrayView<const Scalar> x_(
        PINOCCHIO_EIGEN_CONST_CAST(Vector, x).data(), (size_t)x.size());
    CppAD::cg::ArrayView<const Scalar> w_(
        PINOCCHIO_EIGEN_CONST_CAST(Vector, w).data(), (size_t)w.size());
    CppAD::cg::ArrayView<Scalar> hess_(hess.data(), (size_t)hess.size());

    // auto hess_ = Base::generatedFun_ptr->Hessian(x, 0);
    Base::generatedFun_ptr->Hessian(x_, w_, hess_);
  }

  template <typename StateVector, typename TangentVector>
  void evalHessian(const Eigen::MatrixBase<StateVector> &s,
                   const Eigen::MatrixBase<TangentVector> &u) {
    // fill x
    Eigen::DenseIndex it = 0;
    x.segment(it, nx) = s;
    it += nx;
    x.segment(it, nu) = u;
    it += nu;

    VectorXs w(1);
    w << Scalar(1.0);

    evalHessian(x, w, 0);
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
  ADTangentVectorType ad_u;

  MatrixXs hess;

  VectorXs x;
  VectorXs res;

  const FrameIndex base_idx;
  const pinocchio::SE3Tpl<ADScalar> target;

  inline void cost();
};

template <typename Scalar> void CodeGenCost<Scalar>::cost() {
  pinocchio::framesForwardKinematics(ad_model, ad_data, ad_q);

  ad_f = ADScalar(0);

  auto base_trans = ad_data.oMf[base_idx].translation();
  Eigen::Quaternion<ADScalar> base_rot(ad_data.oMf[base_idx].rotation());

  auto target_trans = target.translation();
  Eigen::Quaternion<ADScalar> target_rot(target.rotation());

  auto lin_delta = base_trans - target_trans;
  ad_f += (lin_delta.transpose() * lin_delta).sum();

  ADScalar ang_dist = base_rot.angularDistance(target_rot);
  ad_f += ang_dist * ang_dist;

  ad_f += (ad_u.transpose() * ad_u).sum();
}

#endif