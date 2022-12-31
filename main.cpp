#include <iostream>
#include <Eigen/Dense>
#include <stdexcept>

#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/scene/scenehandler.h>
#include <pangolin/gl/gldraw.h>

#include <pangolin/utils/timer.h>
#include <opencv2/opencv.hpp>

#include "./frame.h"
#include "./camera_params.hpp"


int main( int /*argc*/, char** /*argv*/ )
{
	pangolin::CreateWindowAndBind("Main",1600,1200);
	glEnable(GL_DEPTH_TEST);

	// Define Projection and initial ModelView matrix
	pangolin::OpenGlRenderState s_cam(
	pangolin::ProjectionMatrix(1600,800,680,680,800,600,0.2,100),
	pangolin::ModelViewLookAt(-2,2,-2, 0,0,0, pangolin::AxisY)
	);

	// Object to hold all scene objects
	pangolin::Renderable scene;

	// Create Interactive View in window
	pangolin::SceneHandler handler(scene,s_cam);
	pangolin::View& d_cam = pangolin::CreateDisplay()
	    .SetBounds(0.0, 1.0, 0.0, 1.0, -1600.0f/800.0f)
	    .SetHandler(&handler);

	d_cam.SetDrawFunction([&](pangolin::View& view){
	view.Activate(s_cam);
	scene.Render();
	});
	
	pangolin::Timer time;

	// Open up video stream
	cv::VideoCapture vid1("./data/vid1_trim.mp4");

	// Safety check
	if (!vid1.isOpened())
	{
		throw std::invalid_argument("Invalid input video stream");
		return -1;
	}

	// Init camera/sensor config
	double w = 960;
	double h = 540;
	double f = (4.7/6.8)*w;
	CameraParams<double> camera(w,h,f);	

	// Store 2 most recent frames for feature matching
	std::queue<cv::Mat> vid_frames;

	// Main program loop
	while( !pangolin::ShouldQuit() )
	{
		// Clear screen and activate view to render into
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		double t = time.Elapsed_ms();
		std::cout << t/1000. << std::endl;

		// Capture frame
		cv::Mat video_frame;
		vid1 >> video_frame;

		if (video_frame.empty()) break;
		vid_frames.push(video_frame);

		// All the action happens inside
		doFrame(scene, camera, vid_frames);

		// Swap frames and Process Events
		pangolin::FinishFrame();

		if (vid_frames.size() > 1) vid_frames.pop();
	}

	cv::waitKey(0);

	vid1.release();

return 0;
}
