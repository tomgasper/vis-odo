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

void poseEstimation(Mat& img_1, Mat& img_2, Mat& R, Mat& t)
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

	std::cout << img_1.type() << std::endl;

	//img_1.convertTo(img_1, CV_32FC1);
	//img_2.convertTo(img_2, CV_32FC1);

	cv::goodFeaturesToTrack(img_1, temp_kpts_1, 300, 0.01, 5,cv::Mat());
	cv::goodFeaturesToTrack(img_2, temp_kpts_2, 300, 0.01, 5, cv::Mat());

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
	for (auto const& i : all_matches)
	{
		for (auto const& j : i)
		{
			if (j.distance <= 10)
				{				
				good_matches.push_back(j);
				}
			}
	}

	Mat matches_img;
	drawMatches(img_1, kpts_1, img_2, kpts_2, good_matches, matches_img);

	// Camera intrinsics
	double w = img_1.size().width;
	double h = img_1.size().height;
	double f = (4.7/6.8)*w;

	Point2d principal_point (w/2.,h/2.);

	Mat K = (Mat_<double>(3,3) << f, 0, w/2.0, 0, f, h/2.0, 0, 0, 1);

	// Calculate the Essential Matrix
	Mat E;
	std::vector<Point2f> match_pts_1,match_pts_2;

	for (int i = 0; i < good_matches.size(); ++i)
	{
		int idx_1 = good_matches[i].queryIdx;
		int idx_2 = good_matches[i].trainIdx;

		match_pts_1.push_back(kpts_1[idx_1].pt);
		match_pts_2.push_back(kpts_2[idx_2].pt);
	}

	E = findEssentialMat(match_pts_1, match_pts_2, f, principal_point);

	std::cout << "ESSENTIAL MATRIX IS: \n" << E << std::endl;

	// Decompose Essential matrix
	Mat s,v,d;
	SVD::compute(E, s, v, d);

	std::cout << "SINGULAR VALUES: \n" << s << std::endl;

	// Recover rotation and translation from Essential Matrix
	recoverPose(E, match_pts_1, match_pts_2, R, t, f, principal_point);	

	imshow("matches", matches_img);
	//waitKey(0);

}

