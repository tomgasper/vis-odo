#include <iostream>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include "./camera_params.hpp"

using namespace cv;

void findKeypoints(Mat& img_1, Mat& img_2, std::vector<cv::Point2f> &match_pts_1, std::vector<cv::Point2f> &match_pts_2)
{
	// Set up ORB detector
	std::vector<KeyPoint> kpts_1, kpts_2;
	Mat desc_1, desc_2;
	Ptr<FeatureDetector> detector = ORB::create();
	Ptr<DescriptorExtractor> descriptor = ORB::create();

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	std::vector<std::vector<DMatch>> all_matches;

	// Temp vector needed to conver from cv::Point2f vector to KeyPoint vector
	std::vector<cv::Point2f> temp_kpts_1,temp_kpts_2;

	cv::goodFeaturesToTrack(img_1, temp_kpts_1, 3000, 0.01, 3 ,cv::Mat());
	cv::goodFeaturesToTrack(img_2, temp_kpts_2, 3000, 0.01, 3, cv::Mat());	

	// Convert found good features to cv keypoints
	for (auto k : temp_kpts_1)
	{
		KeyPoint pt(k, 1.f);
		kpts_1.push_back(pt);
	}

	for (auto k : temp_kpts_2)
	{
		KeyPoint pt(k, 1.f);
		kpts_2.push_back(pt);
	}
	
	// Compute descriptor for each keypoint
	detector->compute(img_1,kpts_1,desc_1);
	detector->compute(img_2,kpts_2,desc_2);

	std::vector<Mat> descs_arr = {desc_1,desc_2};
	std::vector<DMatch> good_matches;

	// Match keypoints
	matcher->knnMatch(desc_1,desc_2,all_matches,2);
	
	// Filter matches by distance
	for (auto const& match : all_matches)
	{
		if (match[0].distance < 0.50*match[1].distance){								
		 good_matches.push_back(match[0]);
		}
	}

	// Point2d principal_point (w/2.,h/2.);

	for (int i = 0; i < good_matches.size(); ++i)
	{
		int idx_1 = good_matches[i].queryIdx;
		int idx_2 = good_matches[i].trainIdx;
	
		match_pts_1.push_back(kpts_1[idx_1].pt);
		match_pts_2.push_back(kpts_2[idx_2].pt);

		// Draw feature points
		cv::circle(img_1, kpts_1[idx_1].pt, 1, cv::Scalar(0,0.0,255.0),2);
		cv::circle(img_1, kpts_2[idx_2].pt, 1, cv::Scalar(0,0.0,255.0),2);

		// Draw lines between previous and current frame
		cv::line(img_1, kpts_1[idx_1].pt, kpts_2[idx_2].pt, cv::Scalar(0,255,0), 2, 4, 0);
	}

	imshow("matches", img_1);
}

template<typename L>
void checkAccuracy(const cv::Mat& R,const cv::Mat& t, const Mat &E, const std::vector<cv::Point2f> &kpts_1, const std::vector<cv::Point2f> &kpts_2, const CameraParams<L>& camera )
{
	// Check if points agree with the epiploar constraint
	
	
	// E should equal t^R -> E = t^R
	cv::Mat t_left_cross = ( cv::Mat_<double>(3,3) << 0, -t.at<double>(2,0), t.at<double>(1,0),
				t.at<double>(2,0), 0 , -t.at<double>(0,0),
				-t.at<double>(1,0), t.at<double>(0,0), 0);

	std::cout << "Essential matrix: " << E << std::endl;
	std::cout << "t^R=" <<'\n' << t_left_cross * R << std::endl;

	cv::Mat Kinv = camera.getInvMat();

	// Epipolar constraints -> x_2.transpose() * E * x_1 = 0
	for ( int i =0; i < kpts_1.size(); i++ )
	{
		Mat pt1 = (Mat_<double>(3,1) << kpts_1[i].x, kpts_1[i].y, 1.);
		Mat pt2 = (Mat_<double>(3,1) << kpts_2[i].x, kpts_2[i].y, 1.);

		pt1 = Kinv * pt1;
		pt2 = Kinv * pt2;

		pt1.at<double>(0,2) = 1.;
		pt2.at<double>(0,2) = 1.;

		Mat dist = pt2.t() * t_left_cross * R * pt1;

		std::cout << "Epipolar constraint = " << dist << std::endl;
	}
}


template<typename L>
void poseEstimation( const std::vector<Point2f> &match_pts_1, const std::vector<Point2f> &match_pts_2, Mat& R, Mat& t, CameraParams<L>& camera)
{
	// Retrive Intrinsic Matrix
	Mat K = camera.getMat();
	Mat Kinv = camera.getInvMat();

	// Calculate the Essential Matrix
	Mat E;
	E = findEssentialMat(match_pts_1, match_pts_2, K, cv::RANSAC, 0.9999, 3.0, 200);	

	// Decompose Essential matrix
	Mat s,v,d;
	SVD::compute(E, s, v, d);

	// Recover rotation and translation from Essential Matrix
	recoverPose(E, match_pts_1, match_pts_2, K, R,t);
}

void triangulate( const std::vector<Point2f> &match_pts_1, const std::vector<Point2f> &match_pts_2,  const cv::Mat &R_in, const cv::Mat &t, CameraParams<double> &camera, std::vector<Eigen::Matrix<double,3,1>> &out_pts)
{
	Mat T_1 = (Mat_<float>(3,4) <<
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0);

	cv::Mat R = R_in;
	
	Mat T_2 = (Mat_<float>(3,4) <<
			R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
			R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
			R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)
		  );

	std::vector<cv::Point2f> pts_1, pts_2;

	cv::Mat K = camera.getMat();

	
	// Normalize points first
	for (int i=0; i < match_pts_1.size(); i++)
	{
		std::vector<float> dist_coeffs;

		

		std::vector<cv::Point2f> pts{match_pts_1[i], match_pts_2[i]};

		cv::undistortPoints(pts, pts, K, dist_coeffs);
 
		pts_1.push_back(pts[0]);
		pts_2.push_back(pts[1]);
	}



	cv::Mat pts_4d_homo;
	cv::triangulatePoints(T_1, T_2, pts_1, pts_2, pts_4d_homo);

	for (int i = 0; i < pts_4d_homo.cols; i++)
	{
		cv::Mat x = pts_4d_homo.col(i);

		// from homo system to unhomo
		x = x / x.at<float>(3,0);

		Eigen::Matrix<double, 3, 1> pt;
	        pt << double(x.at<float>(0,0)), double(x.at<float>(1,0)), double(x.at<float>(2,0));

		pt = pt * 0.1;

		if ( pt(2,0) <= 0.0 ) continue;

		std::cout << pt(2,0) << std::endl;
		out_pts.push_back(pt);
	}

}

// Explicit template instantiation
template void poseEstimation(  const std::vector<Point2f> &match_pts_1, const std::vector<Point2f> &match_pts_2,  Mat& R, Mat& t, CameraParams<double>& camera);
template void checkAccuracy(const cv::Mat& R,const cv::Mat& t, const Mat &E, const std::vector<cv::Point2f> &kpts_1, const std::vector<cv::Point2f> &kpts_2, const CameraParams<double>& camera );

