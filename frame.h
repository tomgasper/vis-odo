#include <Eigen/Dense>
#include <pangolin/scene/scenehandler.h>
#include <opencv2/opencv.hpp>

#include "./camera_params.hpp"
#include "./state.h"

template<typename T>
void visualize(pangolin::Renderable&, const Eigen::Matrix<T,3,3>& , const Eigen::Matrix<T,4,4>& );

template <typename L>
void doFrame(pangolin::Renderable&, CameraParams<L>&, std::queue<cv::Mat>&);
