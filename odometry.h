#include "./camera_params.hpp"
#include <opencv2/core/core.hpp>

void findKeypoints(cv::Mat& img_1, cv::Mat& img_2, std::vector<cv::Point2f> &match_pts_1, std::vector<cv::Point2f> &match_pts_2);

template<typename L>
void poseEstimation( const std::vector<cv::Point2f> &match_pts_1, const std::vector<cv::Point2f> &match_pts_2, cv::Mat& R, cv::Mat& t, CameraParams<L>& camera);

void triangulate( const std::vector<cv::Point2f> &match_pts_1, const std::vector<cv::Point2f> &match_pts_2,  const cv::Mat &R, const cv::Mat &t, CameraParams<double> &camera, std::vector<Eigen::Matrix<double,3,1>> &out_pts);

template<typename L>
void checkAccuracy(const cv::Mat& R,const cv::Mat& t, const cv::Mat &E, const std::vector<cv::Point2f> &kpts_1, const std::vector<cv::Point2f> &kpts_2, const CameraParams<L>& camera );

