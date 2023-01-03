#include "./camera_params.hpp"
#include <opencv2/core/core.hpp>

template<typename L>
void poseEstimation( const std::vector<cv::Point2f> &match_pts_1, const std::vector<cv::Point2f> &match_pts_2, cv::Mat& R, cv::Mat& t, CameraParams<L>& camera);

void findKeypoints(cv::Mat& img_1, cv::Mat& img_2, std::vector<cv::Point2f> &match_pts_1, std::vector<cv::Point2f> &match_pts_2);

void triangulate( const std::vector<cv::Point2f> &match_pts_1, const std::vector<cv::Point2f> &match_pts_2,  const cv::Mat &R, const cv::Mat &t, CameraParams<double> &camera, std::vector<Eigen::Matrix<double,3,1>> &out_pts);
