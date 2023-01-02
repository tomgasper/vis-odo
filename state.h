#include <Eigen/Dense>
#pragma once

struct myState{
	Eigen::Matrix<double,4,4> prevT;

	cv::Mat_<double> w_R;
	cv::Mat_<double> w_t;
};



