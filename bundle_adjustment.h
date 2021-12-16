#ifndef BUNDLE_ADJUSTMENT_HEADER
#define BUNDLE_ADJUSTMENT_HEADER

#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>

using namespace ceres;
using namespace cv;

struct CostFunctor {

	CostFunctor(Mat obs_point3d) :_obs_point3d(obs_point3d) {}

	template <typename T>
	bool operator()(const T* const pose_cw, const T* const map_point3d, T* residual) const {
		T p[3];
		AngleAxisRotationPoint(pose_cw, map_point3d, p);
		p[0] += pose[3];
		p[1] += pose[4];
		p[2] += pose[5];

		double norm = sqrt<T>(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
		T pred_x = p[0] / norm;
		T pred_y = p[1] / norm;
		T pred_z = p[2] / norm;

		T u = T(_obs_point3d.at<T>(0, 0));
		T v = T(_obs_point3d.at<T>(1, 0));
		T w = T(_obs_point3d.at<T>(2, 0));

		T dot = pred_x * u + pred_y * v + pred_z * w;

		residual[0] = 2 * sqrt<T>((1 - dot) / (1 + dot));
		return true;
	}
	Mat _obs_point3d;
};

CostFunction* CreateAutoDiffCostFunction(Mat obs_point3d) {
	return new AutoDiffCostFunction<CostFunctor, 1, 6, 3>(new CostFunctor(obs_point3d));
}

#endif
