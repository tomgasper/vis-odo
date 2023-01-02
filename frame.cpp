#include <iostream>
#include <Eigen/Dense>

#include <pangolin/display/view.h>
#include <pangolin/scene/scenehandler.h>
#include <pangolin/utils/timer.h>
#include "./objs/points.h"
#include "./objs/camera.h"

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <opencv2/core/eigen.hpp>

#include "./odometry.h"
#include "./frame.h"
#include "./camera_params.hpp"

#include "./state.h"

template<typename T>
void visualize(pangolin::Renderable& scene, const Eigen::Matrix<T, 3, 3>& Kinv, const Eigen::Matrix<T,4,4>& T_mat)
{

		// Add new camera to the scene
		auto camera = std::make_shared<pangolin::Camera<double>>(Kinv, 960. ,540.  , 0.2f, 1);
		camera->T_pc = pangolin::OpenGlMatrix(T_mat);
		scene.Add(camera);

		// temp point generation for visualization
		std::vector<Eigen::Matrix<double, 3, 1>> pts;
		std::random_device rand;
		std::mt19937 gen(rand());
		std::uniform_int_distribution<> distr(-100,100);
		for (int i = -10; i < 10; i++)
		{

			Eigen::Matrix<double,3,1> pt = {distr(gen)/100., distr(gen)/100.,distr(gen)/100.};
			pts.push_back(pt);
		}

		// Add points to the scene
		auto points = std::make_shared<pangolin::Points<double>>(pts);
		points->T_pc = pangolin::OpenGlMatrix(T_mat);
		scene.Add(points);
}

template<typename L>
void doFrame(pangolin::Renderable& scene,const CameraParams<L>& camera,  std::queue<cv::Mat>& vid_frames, struct myState& state)
{
		// Not enough frames for feature tracking, 2 are needed
		if (vid_frames.size() < 2) return;

		// Process frames from the video stream
		cv::Mat img1 = vid_frames.back();
		cv::Mat img2 = vid_frames.front();

		cv::resize(img1,img1, cv::Size(), 0.5,0.5);
		cv::resize(img2, img2, cv::Size(), 0.5,0.5);

		cv::cvtColor(img1,img1, cv::COLOR_BGR2GRAY);
		cv::cvtColor(img2, img2, cv::COLOR_BGR2GRAY);

		// Init matrices for the poseEstimation output
		cv::Mat R, t;

		
		
		// Feed new frame to the pose estimation
		poseEstimation(img1,img2, R, t);

		// std::cout << "R: " << R << std::endl;
		std::cout << "t: " << t << std::endl;

		// state.w_t = state.w_t + 0.1 * (state.w_R * t);
		// state.w_R = R.() * state.w_R;

		// R = state.w_R;
		// t = state.w_t;
		//
		//
	

		/*
		Eigen::Matrix4d T_mat;
		T_mat << R.at<double>(0,0),R.at<double>(0,1),R.at<double>(0,2), t.at<double>(0,0),
			 R.at<double>(1,0),R.at<double>(1,1),R.at<double>(1,2), t.at<double>(1,0),
			 R.at<double>(2,0),R.at<double>(2,1),R.at<double>(2,2), t.at<double>(2,0),
			 0,0,0,1;
		*/


		Eigen::Matrix4d T_mat;
		T_mat << R.at<double>(0,0),R.at<double>(0,1),R.at<double>(0,2), 0.1*t.at<double>(0,0),
			 R.at<double>(1,0),R.at<double>(1,1),R.at<double>(1,2), 0.1*t.at<double>(1,0),
			 R.at<double>(2,0),R.at<double>(2,1),R.at<double>(2,2), 0.1*t.at<double>(2,0),
			 0,0,0,1;

		Eigen::Matrix4d flip_Y;
		flip_Y << 1, 0, 0, 0,
		          0, -1, 0, 0,
		          0, 0, -1, 0,
		          0, 0, 0, 1;


		
		// cv::Mat t_norm = t / cv::norm(t);
		// cv::Mat t_to_cam = -(R.t() * t_norm);
		//

		/*
		T_mat << 1, 0, 0, t.at<double>(0,0),
		      	0, 1, 0 , t.at<double>(1,0),
			0, 0, 1, t.at<double>(2,0),
			0, 0, 0, 1;

		*/

		// new transform
		state.prevT = (state.prevT * T_mat);

		Eigen::Matrix4d flipMat = state.prevT.transpose() * flip_Y;
		Eigen::Matrix4d fin = flipMat.transpose();

		visualize(scene, camera.getInvMatEig(), fin);

		}

template void doFrame(pangolin::Renderable& scene, const CameraParams<double>& camera,  std::queue<cv::Mat>& vid_frames, struct myState& state);

