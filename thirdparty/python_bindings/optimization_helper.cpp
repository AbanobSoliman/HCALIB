/*
    This code is developed by Abanob Soliman, PhD Candidate, IBISC Laboratory
    As part of PhD Thesis, 2022
*/

#include "optimization_helper.h"

namespace py = pybind11;
void add_custom_cost_functions(py::module &m);

// 1- RGB Frames Features Structured Reprojection Errors Factor
struct ReprojectionErrors {
  ReprojectionErrors(const Eigen::Vector2d &corner, const Eigen::Matrix2d &Q) : corner(corner), Q(Q) {}
  template<typename T>
  bool operator()(const T* T_w_i, const T* T_i_c, const T* camera, const T* point, T* residuals) const {
    Eigen::Vector<T, 2> predicted;  // Reprojected P3D on RGB 2D frame after radtan distortion
    OptimHelp::ProjectP3DtoP2D(T_w_i, T_i_c, camera, point, predicted);
    // The error is the difference between the predicted and observed corner on RGB Frame.
    Eigen::Map<Matrix<T,2,1>> r_vec(residuals);
    r_vec = Q.inverse() * (predicted - corner.cast<T>());
    return true;
  }
  static ceres::CostFunction *Create(const Eigen::Vector2d &corner, const Eigen::Matrix2d &Q) {
    return (new ceres::AutoDiffCostFunction<ReprojectionErrors, 2, 7, 7, 10, 3>(new ReprojectionErrors(corner, Q)));
  }
  Eigen::Vector2d corner;
  Eigen::Matrix2d Q;
};

// 2- Global Point Cloud Factor (Stereo Vision Concept)
struct GlobalCloudOpt {
  GlobalCloudOpt(const Eigen::Vector2d &corner, const Eigen::Matrix2d &Q) : corner(corner), Q(Q) {}
  template<typename T>
  bool operator()(const T* T_w_i, const T* T_i_c, const T* T_d_c, const T* camera, const T* dcamera, const T* point, T* residuals) const {
    Eigen::Matrix<T, 2, 1> p2d_d, p2d_c;
    Eigen::Matrix<T, 2, 1>const cornerC(corner.cast<T>());
    OptimHelp::ProjectP3DepthDist(T_w_i, T_i_c, T_d_c, dcamera, point, p2d_d);
    OptimHelp::ProjectP2DCamDist(T_w_i, T_i_c, T_d_c, camera, dcamera, cornerC, p2d_c);
    // The error is the difference between the predicted and observed corner on Depth Frame.
    Eigen::Map<Matrix<T,2,1>> r_vec(residuals);
    r_vec = Q.inverse() * (p2d_d - p2d_c);
    return true;
  }
  static ceres::CostFunction *Create(const Eigen::Vector2d &corner, const Eigen::Matrix2d &Q) {
    return (new ceres::AutoDiffCostFunction<GlobalCloudOpt, 2, 7, 7, 7, 10, 9, 3>(new GlobalCloudOpt(corner, Q)));
  }
  Eigen::Vector2d corner;
  Eigen::Matrix2d Q;
};

// 3- IMU Preintegrated Factor between 2 RGB frames i,j
struct IMUFactor {
  IMUFactor(const Eigen::MatrixXd &imu_meas, const Eigen::Matrix<double, 6, 6> &Q, const double dt, const Eigen::Matrix<double, 3, 1> &grav, const Eigen::Matrix<double, 6, 1> &bias_i) : imu_meas(imu_meas), Q(Q), dt(dt), grav(grav), bias_i(bias_i) {}
  template<typename T>
  bool operator()(const T* _Twi_i, const T* _Twi_j, const T* vel_i, const T* vel_j, const T* imustates_i, T* residuals) const {
    Eigen::Matrix<T,3,3> I3(Eigen::Matrix<T,3,3>::Identity()); // I3
    Eigen::Matrix<T,3,3> O3(Eigen::Matrix<T,3,3>::Zero()); // 03
    // Extracting the Rwi_i,j and twi_i,j
    SE3<T>const Twi_i(_Twi_i);
    SE3<T>const Twi_j(_Twi_j);
    Eigen::Matrix<T,3,3>const Rwi_i(Twi_i.q_.R());   // Rwi_i here
    Eigen::Map<Matrix<T,3,1>>const twi_i(Twi_i.t_);  // twi_i here
    Eigen::Matrix<T,3,3>const Rwi_j(Twi_j.q_.R());   // Rwi_j here
    Eigen::Map<Matrix<T,3,1>>const twi_j(Twi_j.t_);  // twi_j here
    // Extracting the v_i,j
    Eigen::Matrix<T,3,1>const v_i(vel_i[0],vel_i[1],vel_i[2]);
    Eigen::Matrix<T,3,1>const v_j(vel_j[0],vel_j[1],vel_j[2]);
    // Extracting IMU States 7 -1(tic[6]) parameters
    Eigen::Matrix<T,3,1>const bw(imustates_i[0],imustates_i[1],imustates_i[2]); // b_w_hat (state)
    Eigen::Matrix<T,3,1>const ba(imustates_i[3],imustates_i[4],imustates_i[5]); // b_a_hat (state)
    Eigen::Matrix<T,3,1>const bwi(bias_i(seq(0,2), 0).cast<T>()); // b_w_bar (given)
    Eigen::Matrix<T,3,1>const bai(bias_i(seq(3,5), 0).cast<T>()); // b_a_bar (given)
    Eigen::Matrix<T,3,1>const g(grav(seq(0,2), 0).cast<T>());
    Eigen::Matrix<T,6,6>const Q_eta(Q.cast<T>());  // cov_b_wa
    Eigen::Matrix<T,9,9> A(Eigen::Matrix<T,9,9>::Zero());  // A
    Eigen::Matrix<T,9,6> B(Eigen::Matrix<T,9,6>::Zero());  // B
    Eigen::Matrix<T,9,9> Q_pre(Eigen::Matrix<T,9,9>::Zero());  // Preintegrated Covariance
    Eigen::Matrix<T,3,1> wkp1,wJr; 
    Eigen::Matrix<T,3,3> Jr(O3);     // Initially set with 03
    // Pre-Integrating the dR_tilda, dv_tilda, dp_tilda
    Eigen::Matrix<T,3,3> dR_tilda(I3); // Initially set with I3    
    Eigen::Matrix<T,3,1> dv_tilda(T(0.0),T(0.0),T(0.0));             // Initially set with 0
    Eigen::Matrix<T,3,1> dp_tilda(T(0.0),T(0.0),T(0.0));             // Initially set with 0
    Eigen::Matrix<T,3,1> a,w;
    T dt_ij = T(0.0);
    // R,v,p,Q propagation
    for (int k=0; k<=imu_meas.rows(); ++k){
    	w = imu_meas(k,seq(0,2)).transpose().cast<T>();
    	a = imu_meas(k,seq(3,5)).transpose().cast<T>();
    	dv_tilda += dR_tilda * (a - ba) * T(dt);
    	dp_tilda += T(1.5) * dR_tilda * (a - ba) * T(dt) * T(dt);
    	wJr = w - bw;
    	OptimHelp::rightJacobian(wJr, Jr);
    	A << SO3<T>::Exp((w - bw) * T(dt)).R().transpose(), O3, O3,
    	     -dR_tilda * SO3<T>::hat(a) * T(dt), I3, O3,
    	     -T(0.5) * dR_tilda * SO3<T>::hat(a) * T(dt) * T(dt), I3 * T(dt), I3;
    	B << Jr * T(dt), O3,
    	     O3, dR_tilda * T(dt),
    	     O3, T(0.5) * dR_tilda * T(dt) * T(dt);
    	Q_pre = A * Q_pre * A.transpose() + B * Q_eta * B.transpose();
    	dR_tilda *= SO3<T>::Exp((w - bw) * T(dt)).R();
    	dt_ij += T(dt);
    }
    // Calculate Jacobians to dbw,a   
    Eigen::Matrix<T,3,3> dR(I3); // Initially set with I3  
    Eigen::Matrix<T,3,3> dr(I3); // Initially set with I3    
    Eigen::Matrix<T,3,3> dRdbw(O3);  // Initially set with 03   
    Eigen::Matrix<T,3,3> dvdba(O3);  // Initially set with 03   
    Eigen::Matrix<T,3,3> dvdbw(O3);  // Initially set with 03   
    Eigen::Matrix<T,3,3> dpdba(O3);  // Initially set with 03   
    Eigen::Matrix<T,3,3> dpdbw(O3);  // Initially set with 03  
    Eigen::Matrix<T,3,1> dbw(bwi - bw);  // db_w = b_w_hat - b_w_bar
    Eigen::Matrix<T,3,1> dba(bai - ba);  // db_a = b_a_hat - b_a_bar 
    for (int k=0; k<=imu_meas.rows(); ++k){
    	w = imu_meas(k,seq(0,2)).transpose().cast<T>();
    	dR *= SO3<T>::Exp((w - bw) * T(dt)).R();
    	dr = dR;
    	for (int l=k+1; l<=imu_meas.rows(); ++l){
    	    wkp1 = imu_meas(l,seq(0,2)).transpose().cast<T>();
    	    dr *= SO3<T>::Exp((wkp1 - bw) * T(dt)).R();
    	}
    	wJr = w - bw;
    	OptimHelp::rightJacobian(wJr, Jr);
    	dRdbw -= dr.transpose() * Jr * T(dt);
    }
    for (int k=0; k<=imu_meas.rows(); ++k){
        a = imu_meas(k,seq(3,5)).transpose().cast<T>();
    	dvdba += -(dR_tilda * T(dt));
    	dvdbw += -(dR_tilda * SO3<T>::hat(a - ba) * dRdbw * T(dt));
	dpdba += -(T(1.5) * dR_tilda * T(dt) * T(dt));
	dpdbw += -(T(1.5) * dR_tilda * SO3<T>::hat(a - ba) * dRdbw * T(dt) * T(dt));
    }
    // Calculate accumulated DR_tilda, Dv_tilda, Dp_tilda
    Eigen::Matrix<T,3,3> DR_tilda(dR_tilda * SO3<T>::Exp(dRdbw*dbw).R());
    Eigen::Matrix<T,3,1> Dv_tilda(dv_tilda + dvdbw * dbw + dvdba * dba);
    Eigen::Matrix<T,3,1> Dp_tilda(dp_tilda + dpdbw * dbw + dpdba * dba);
    // Calculating the Residuals of current IMU stack
    Eigen::Map<Matrix<T,9,1>> r_vec(residuals);
    r_vec(seq(0,2)) = SO3<T>::Log(SO3<T>::fromR(DR_tilda.transpose() * (Rwi_i.transpose() * Rwi_j))); // delta_R
    r_vec(seq(3,5)) = Rwi_i.transpose() * (v_j - v_i - g * T(dt_ij)) - Dv_tilda; // delta_v
    r_vec(seq(6,8)) = Rwi_i.transpose() * (twi_j - twi_i - v_i * T(dt_ij) - T(0.5) * g * T(dt_ij) * T(dt_ij)) - Dp_tilda; // delta_p
    r_vec = Q_pre.inverse() * r_vec;
    return true;
  }
  static ceres::CostFunction *Create(const Eigen::MatrixXd &imu_meas, const Eigen::Matrix<double, 6, 6> &Q, const double dt, const Eigen::Matrix<double, 3, 1> &grav, const Eigen::Matrix<double, 6, 1> &bias_i) {
    return (new ceres::AutoDiffCostFunction<IMUFactor, 9, 7, 7, 3, 3, 7>(new IMUFactor(imu_meas, Q, dt, grav, bias_i)));
  }
  Eigen::MatrixXd imu_meas;
  Eigen::Matrix<double, 6, 6> Q;
  double dt;
  Eigen::Matrix<double, 3, 1> grav;
  Eigen::Matrix<double, 6, 1> bias_i;
};

// 4- IMU Bias Factor
struct IMUBiasFactor {
  IMUBiasFactor(const Eigen::Matrix3d &Q_w, const Eigen::Matrix3d &Q_a) : Q_w(Q_w), Q_a(Q_a) {}
  template<typename T>
  bool operator()(const T* imustates_i, const T* imustates_j, T* residuals) const {
    // Extracting IMU biases parameters
    Eigen::Matrix<T, 3, 1>const bw_i(imustates_i[0],imustates_i[1],imustates_i[2]);
    Eigen::Matrix<T, 3, 1>const ba_i(imustates_i[3],imustates_i[4],imustates_i[5]);
    Eigen::Matrix<T, 3, 1>const bw_j(imustates_j[0],imustates_j[1],imustates_j[2]);
    Eigen::Matrix<T, 3, 1>const ba_j(imustates_j[3],imustates_j[4],imustates_j[5]);
    // Calculating the Residuals of current and previous IMU Bias states
    Eigen::Map<Matrix<T,6,1>> r_vec(residuals);
    r_vec(seq(0,2)) = Q_w.inverse() * (bw_j - bw_i); // delta_bw_ij
    r_vec(seq(3,5)) = Q_a.inverse() * (ba_j - ba_i); // delta_ba_ij
    return true;
  }
  static ceres::CostFunction *Create(const Eigen::Matrix3d &Q_w, const Eigen::Matrix3d &Q_a) {
    return (new ceres::AutoDiffCostFunction<IMUBiasFactor, 6, 7, 7>(new IMUBiasFactor(Q_w, Q_a)));
  }
  Eigen::Matrix3d Q_w;
  Eigen::Matrix3d Q_a;
};

// 5- IMU tic Factor
struct IMUticFactor {
  IMUticFactor(const double tcam, const double timu) : tcam(tcam), timu(timu) {}
  template<typename T>
  bool operator()(const T* imustates_i, T* residuals) const {
    // Calculating the Residual
    const T &tic = imustates_i[6];
    residuals[0] = T(tcam - (timu + tic));
    return true;
  }
  static ceres::CostFunction *Create(const double tcam, const double timu) {
    return (new ceres::AutoDiffCostFunction<IMUticFactor, 1, 7>(new IMUticFactor(tcam, timu)));
  }
  double tcam;
  double timu;
};

// 6- Pose Graph Optimizer Factor using GPS readings
struct PoseGraphOpt {
  PoseGraphOpt(const Eigen::Matrix<double,7,1> &X_vec, const Eigen::Matrix<double,6,6> &Q) : Xij_(X_vec), Q_inv_(Q.inverse()) {}
  template<typename T>
  bool operator()(const T* _Xi_hat, const T* _Xj_hat, T* _res) const
  {
    SE3<T> Xi_hat(_Xi_hat);
    SE3<T> Xj_hat(_Xj_hat);
    Eigen::Map<Matrix<T,6,1>> r(_res);
    r = Q_inv_ * (Xi_hat.inverse() * Xj_hat - Xij_.cast<T>());  
    return true;
  }
  static ceres::CostFunction *Create(const Eigen::Matrix<double,7,1> &Xij, const Eigen::Matrix<double,6,6> &Q) {
    return new ceres::AutoDiffCostFunction<PoseGraphOpt,
                                           6,
                                           7,
                                           7>(new PoseGraphOpt(Xij, Q));
  }
  SE3d Xij_;
  Eigen::Matrix<double,6,6> Q_inv_;
};

// 7- Range Factor using GPS readings
struct ScaleFactor {
  ScaleFactor(double &rij, double &qij)
  {
    rij_ = rij;
    qij_inv_ = 1.0 / qij;
  }
  template<typename T>
  bool operator()(const T* _Xi_hat, const T* _Xj_hat, T* _res) const
  {
    Eigen::Matrix<T,3,1> ti_hat(_Xi_hat), tj_hat(_Xj_hat);
    *_res = static_cast<T>(qij_inv_) * (static_cast<T>(rij_) - (tj_hat - ti_hat).norm());
    return true;
  }
  static ceres::CostFunction *Create(double &rij, double &qij) {
    return new ceres::AutoDiffCostFunction<ScaleFactor,
                                           1,
                                           7,
                                           7>(new ScaleFactor(rij, qij));
  }
  double rij_;
  double qij_inv_;
};

void add_custom_cost_functions(py::module &m) {
  // Use pybind11 code to wrap the cost functor as defined in C++s
  m.def("ReprojectionErrors", &ReprojectionErrors::Create);
  m.def("GlobalCloudOpt", &GlobalCloudOpt::Create);
  m.def("IMUFactor", &IMUFactor::Create);
  m.def("IMUBiasFactor", &IMUBiasFactor::Create);
  m.def("IMUticFactor", &IMUticFactor::Create);
  m.def("PoseGraphOpt", &PoseGraphOpt::Create);
  m.def("ScaleFactor", &ScaleFactor::Create);
}
