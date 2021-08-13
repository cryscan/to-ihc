#ifndef __DYNAMICS_HPP__
#define __DYNAMICS_HPP__

#include <pinocchio/codegen/code-generator-base.hpp>

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/cholesky.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea.hpp>

template <typename Scalar>
struct CodeGenDynamics : public pinocchio::CodeGenBase<Scalar> {
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

  CodeGenDynamics(const Model &model, Scalar dt, int num_iters, Scalar mu,
                  const std::vector<FrameIndex> &ee_idx,
                  const std::string &function_name = "dynamics",
                  const std::string &library_name = "cg_dynamics_eval")
      : Base(model, model.nq + 2 * model.nv - 6, model.nq + model.nv,
             function_name, library_name),
        nx(model.nq + model.nv), nu(model.nv - 6), dt(dt), num_iters(num_iters),
        mu(mu), num_ees(ee_idx.size()), ee_idx(ee_idx) {
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

    Eigen::DenseIndex it = 0;
    ad_q = ad_X.segment(it, ad_model.nq);
    it += ad_model.nq;
    ad_v = ad_X.segment(it, ad_model.nv);
    it += ad_model.nv;
    ad_u = ad_X.segment(it, nu);
    it += nu;

#ifdef DYNAMICS_IMPL
    ADConfigVectorType qe(ad_model.nq);
    ADTangentVectorType ve(ad_model.nv);
    step(qe, ve);

    it = 0;
    ad_Y.segment(it, ad_model.nq) = qe;
    it += ad_model.nq;
    ad_Y.segment(it, ad_model.nv) = ve;
    it += ad_model.nv;
#endif

    ad_fun.Dependent(ad_X, ad_Y);
    ad_fun.optimize("no_compare_op");
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
    f = res = Base::y;
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
    df_dx = Base::jac.middleCols(it, nx);
    it += nx;
    df_du = Base::jac.middleCols(it, nu);
    it += nu;
  }

  const int nx;
  const int nu;

  const size_t num_ees;
  const std::vector<FrameIndex> ee_idx;

  VectorXs f;
  MatrixXs df_dx, df_du;

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

  const Scalar dt;
  const size_t num_iters;
  const Scalar mu;

  inline ADVectorXs prox(const ADVectorXs &p);
  inline void step(ADConfigVectorType &qe, ADTangentVectorType &ve);
};

#ifdef DYNAMICS_IMPL

template <typename Scalar>
typename CodeGenDynamics<Scalar>::ADVectorXs
CodeGenDynamics<Scalar>::prox(const ADVectorXs &p) {
  auto pn = CppAD::max(p(2), ADScalar(0));
  auto pt0 = CppAD::max(CppAD::min(p(0), pn * mu), -pn * mu);
  auto pt1 = CppAD::max(CppAD::min(p(1), pn * mu), -pn * mu);

  ADVectorXs res(3);
  res << pt0, pt1, pn;
  return res;
}

template <typename Scalar>
void CodeGenDynamics<Scalar>::step(ADConfigVectorType &qe,
                                   ADTangentVectorType &ve) {
  auto qm = pinocchio::integrate(ad_model, ad_q, ad_v * (dt / 2));

  // pinocchio::computeMinverse(ad_model, ad_data, qm);
  pinocchio::crba(ad_model, ad_data, qm);
  pinocchio::cholesky::decompose(ad_model, ad_data);
  pinocchio::cholesky::computeMinv(ad_model, ad_data);

  ADTangentVectorType tau(ad_model.nv);
  tau.setZero();
  tau.segment(6, nu) = ad_u;

  pinocchio::nonLinearEffects(ad_model, ad_data, qm, ad_v);
  auto h = tau - ad_data.nle;

  // collision detection and resolution
  ADScalar zero(0), one(1);

  // assemble contact jacobians
  pinocchio::computeJointJacobians(ad_model, ad_data, qm);
  ADMatrixXs J = ADMatrixXs::Zero(3 * num_ees, ad_model.nv);

  for (size_t i = 0; i < num_ees; i++) {
    ADMatrixXs j = ADMatrixXs::Zero(6, ad_model.nv);
    pinocchio::getFrameJacobian(ad_model, ad_data, ee_idx[i],
                                pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
                                j);
    J.template middleRows<3>(i * 3) = j.template topRows<3>();
  }

  auto m_inv_h = ad_data.Minv * h;
  auto m_inv_Jt = ad_data.Minv * J.transpose();

  auto G = J * m_inv_Jt;
  auto c = J * ad_v + J * m_inv_h * dt;

  ADVectorXs r(num_ees);
  ADVectorXs b(num_ees);
  ADVectorXs p(3 * num_ees);

  pinocchio::framesForwardKinematics(ad_model, ad_data, qm);
  pinocchio::computeTotalMass(ad_model, ad_data);

  // compute b
  for (size_t i = 0; i < num_ees; i++) {
    auto t = ad_data.oMf[ee_idx[i]].translation();
    b(i) = CppAD::CondExpLe(t(2), zero, one, zero);
  }

  for (size_t i = 0; i < num_ees; i++) {
    // init p
    auto p_i = p.template segment<3>(3 * i);
    p_i = -ad_data.mass[0] * dt * ad_model.gravity.linear() /
          CppAD::max(b.sum(), one);

    // compute r
    for (size_t j = 0; j < num_ees; j++) {
      auto G_ij = G.template block<3, 3>(3 * i, 3 * j);
      r(i) += G_ij.determinant();
    }
    r(i) = Scalar(1) / CppAD::max(r(i), ADScalar(0.001));
  }

  for (size_t k = 0; k < num_iters; k++) {
    for (size_t i = 0; i < num_ees; i++) {
      auto p_i = p.template segment<3>(3 * i);
      auto c_i = c.template segment<3>(3 * i);
      auto r_i = r(i);

      for (size_t j = 0; j < num_ees; j++) {
        auto G_ij = G.template block<3, 3>(3 * i, 3 * j);
        auto p_j = p.template segment<3>(3 * j);
        p_i -= b(j) * r_i * (G_ij * p_j + c_i);
      }

      p_i = b(i) * prox(p_i);
    }
  }

  ve = ad_v + m_inv_h * dt + m_inv_Jt * p;
  qe = pinocchio::integrate(ad_model, ad_q, (ad_v + ve) * (dt / 2));

  // correction by IK
  for (size_t i = 0; i < num_ees; i++) {
    pinocchio::framesForwardKinematics(ad_model, ad_data, qe);
    pinocchio::computeJointJacobians(ad_model, ad_data, qe);
    auto t = ad_data.oMf[ee_idx[i]].translation();

    ADVectorXs d(3);
    d << zero, zero, -t(2);

    ADMatrixXs j(6, ad_model.nv);
    pinocchio::getFrameJacobian(ad_model, ad_data, ee_idx[i],
                                pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
                                j);
    auto J = j.template topRows<3>();

    ADVectorXs v = J.colPivHouseholderQr().solve(d);
    pinocchio::integrate(ad_model, qe, b(i) * v, qe);
  }
}

#endif

#endif