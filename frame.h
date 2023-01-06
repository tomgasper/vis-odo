#include <Eigen/Dense>
#include <pangolin/scene/scenehandler.h>
#include <opencv2/opencv.hpp>

#include "./camera_params.hpp"


template<typename T>
void visualize(pangolin::Renderable&, const std::vector<Eigen::Matrix<T,3,1>> &pts, const Eigen::Matrix<T,3,3>& , const Eigen::Matrix<T,4,4>& );

template <typename L>
void doFrame(pangolin::Renderable&, CameraParams<L>&, std::queue<cv::Mat>&);
