#include <iostream>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;

void poseEstimation( Mat& img_1, Mat& img_2, Mat& R, Mat& t)
{
	// Set up ORB detector
	std::vector<KeyPoint> kpts_1, kpts_2;
	Mat desc_1, desc_2;
	Ptr<FeatureDetector> detector = ORB::create();
	Ptr<DescriptorExtractor> descriptor = ORB::create();

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	std::vector<std::vector<DMatch>> all_matches;

	// kpts temp arr
	std::vector<cv::Point2f> temp_kpts_1,temp_kpts_2;

	cv::goodFeaturesToTrack(img_1, temp_kpts_1, 3000, 0.01, 3,cv::Mat());
	cv::goodFeaturesToTrack(img_2, temp_kpts_2, 3000, 0.01, 3, cv::Mat());

	//detector->detect(img_1,kpts_1);
	//detector->detect(img_2,kpts_2);
	//
	
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

	detector->compute(img_1,kpts_1,desc_1);
	detector->compute(img_2,kpts_2,desc_2);

	std::vector<Mat> descs_arr = {desc_1,desc_2};
	std::vector<DMatch> good_matches;

	matcher->knnMatch(desc_1,desc_2,all_matches,2);

	
	// Filter matches by distance
	for (auto const& match : all_matches)
	{
		if (match[0].distance < 0.5*match[1].distance){								
		 good_matches.push_back(match[0]);
		}
	}

	Mat matches_img = img_2;

	// Camera intrinsics
	double w = img_1.size().width;
	double h = img_1.size().height;
	double f = (4.7/6.8)*w;

	Point2d principal_point (w/2.,h/2.);

	Mat K = (Mat_<double>(3,3) <<   f, 0, w/2.0,
		       			0, f, h/2.0,
					0, 0, 1);
	Mat Kinv = K.inv();

	// Calculate the Essential Matrix
	Mat E;
	std::vector<Point2f> match_pts_1,match_pts_2;

	for (int i = 0; i < good_matches.size(); ++i)
	{
		int idx_1 = good_matches[i].queryIdx;
		int idx_2 = good_matches[i].trainIdx;
	

		match_pts_1.push_back(kpts_1[idx_1].pt);
		match_pts_2.push_back(kpts_2[idx_2].pt);
		cv::circle(img_1, kpts_1[idx_1].pt, 1, cv::Scalar(0,0.0,255.0),2);
		cv::circle(img_1, kpts_2[idx_2].pt, 1, cv::Scalar(0,0.0,255.0),2);


		cv::line(img_1, kpts_1[idx_1].pt, kpts_2[idx_2].pt, cv::Scalar(0,255,0), 2, 4, 0);
	}
	

	// std::cout << "norm points" << match_pts_1 << std::endl;

	E = findEssentialMat(match_pts_1, match_pts_2, K, cv::RANSAC, 0.9999, 3.0, 200);	

	// Decompose Essential matrix
	Mat s,v,d;
	SVD::compute(E, s, v, d);

	// Recover rotation and translation from Essential Matrix
	recoverPose(E, match_pts_1, match_pts_2, K, R,t);


	// Check accuracy
	Mat t_left_cross = ( Mat_<double>(3,3) << 0, -t.at<double>(2,0), t.at<double>(1,0),
				t.at<double>(2,0), 0 , -t.at<double>(0,0),
				-t.at<double>(1,0), t.at<double>(0,0), 0);

	// std::cout << "essential matrix: " << E << std::endl;
	// std::cout << "t^R=" <<'\n' << t_left_cross * R << std::endl;

	// check epipolar constraints
	for ( auto const& m : good_matches )
	{



		Mat pt1 = (Mat_<double>(3,1) << kpts_1[m.queryIdx].pt.x, kpts_1[m.queryIdx].pt.y, 1.);
		Mat pt2 = (Mat_<double>(3,1) << kpts_2[m.trainIdx].pt.x, kpts_2[m.trainIdx].pt.y, 1.);

		pt1 = Kinv * pt1;
		pt2 = Kinv * pt2;

		pt1.at<double>(0,2) = 1.;
		pt2.at<double>(0,2) = 1.;

		Mat dist = pt2.t() * t_left_cross * R * pt1;
		//std::cout << "epipolar constraint = " << dist << std::endl;
	}

	

	imshow("matches", img_1);
	// waitKey(0);

}

