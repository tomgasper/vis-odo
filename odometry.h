#include "./camera_params.hpp"
#include <opencv2/core/core.hpp>

template<typename L>
void poseEstimation(cv::Mat& img_1, cv::Mat& img_2, cv::Mat& R, cv::Mat& t, CameraParams<L>&);
