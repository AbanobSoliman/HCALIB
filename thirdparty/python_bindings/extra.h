/*
template <typename T>
inline void TransformW2F(const T q[7], const T pt[3], Eigen::Vector<T, 3>& result) {
  // camera[0,1,2,3] are the angle-axis rotation quaternion rotations qw,qx,qy,qz.
  //camera[4,5,6] are the translation.
  const Eigen::Vector<T, 3> t_cam(q[4],q[5],q[6]);
  // 'scale' is 1 / norm(q).
  const T scale =
      T(1) / sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);

  // Make unit-norm version of q.
  const T qu[4] = {
      scale * q[0],
      scale * q[1],
      scale * q[2],
      scale * q[3],
  };

  // clang-format off
  T uv0 = qu[2] * pt[2] - qu[3] * pt[1];
  T uv1 = qu[3] * pt[0] - qu[1] * pt[2];
  T uv2 = qu[1] * pt[1] - qu[2] * pt[0];
  uv0 += uv0;
  uv1 += uv1;
  uv2 += uv2;
  result[0] = pt[0] + qu[0] * uv0;
  result[1] = pt[1] + qu[0] * uv1;
  result[2] = pt[2] + qu[0] * uv2;
  result[0] += qu[2] * uv2 - qu[3] * uv1;
  result[1] += qu[3] * uv0 - qu[1] * uv2;
  result[2] += qu[1] * uv1 - qu[2] * uv0;
  
  result += t_cam;
  //std::cout<< result<<std::endl;
  // clang-format on
}
*/
