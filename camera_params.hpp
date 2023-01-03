#include <Eigen/Dense>
#include <stdexcept>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#pragma once

template<typename T>
class CameraParams
{
public:
	CameraParams(T w, T h, T f)
	{
		if (w <= 0 || h <= 0 || f <= 0) throw std::invalid_argument("Invalid camera parameters");

		_w = w;
		_h = h;
		_f = f;
		_principal_pt = std::make_pair(w/2,h/2.);

		// Init Transform state
		_TMat = Eigen::Matrix<T, 4, 4>::Identity();
		
		// Insert the data into matrix form
		initMatrix();
		initInvMatrix(_K);
	}

	const cv::Mat_<T>& getMat() const
	{
		return _K;
	}

	const Eigen::Matrix<T, 3, 3>& getMatEig() const
	{
		return _Eig_K;
	}

	const cv::Mat_<T>& getInvMat() const
	{
		return _Kinv;
	}

	const Eigen::Matrix<T, 3, 3>& getInvMatEig() const
	{
		return _Eig_Kinv;
	}

	const std::pair<T,T>& getPrincipalPt() const
	{
		return _principal_pt;
	}

	void setTransformMat(Eigen::Matrix<T,4,4> TMat)
	{
		_TMat = TMat;
	}

	const Eigen::Matrix<T,4,4>& getTransformMatEig() const
	{
		return _TMat;
	}

private:
	void initMatrix()
	{
		_K = (cv::Mat_<T>(3,3) << _f, 0, _w/2.0, 0, _f, _h/2.0, 0, 0, 1);
		cv::cv2eigen(_K, _Eig_K);
	}

	void initInvMatrix(const cv::Mat_<T>& K)
	{
		_Kinv = K.inv();
		cv::cv2eigen(_Kinv, _Eig_Kinv);
	}

	T _w;
	T _h;
	T _f;
	std::pair<T,T> _principal_pt;

	// Store both OpenCv and Eigen mats to avoid unnecessary conversion later on 
	cv::Mat_<T> _K;
	cv::Mat_<T> _Kinv;
	Eigen::Matrix<T, 3, 3> _Eig_K;
	Eigen::Matrix<T, 3, 3> _Eig_Kinv;

	// Transform matrix
	
	Eigen::Matrix<T, 4,4> _TMat;
	
};

