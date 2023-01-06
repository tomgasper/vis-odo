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


int main( int argc, char *argv[]  )
{
	// Read command line arguments
	double INPUT_f;
	int INPUT_skip = 5;
	std::string INPUT_vid_location;
	
	if (argc == 4)
	{
		std::istringstream iss(argv[1]);
		iss >> INPUT_f;

		std::istringstream iss2(argv[2]);
		iss2 >> INPUT_skip;

		std::istringstream iss3(argv[3]);
		iss3 >> INPUT_vid_location;

	} else std::invalid_argument("Invalid number of command line arguments");


	pangolin::CreateWindowAndBind("Main",1600,1000);
	glEnable(GL_DEPTH_TEST);

	// Define Projection and initial ModelView matrix
	pangolin::OpenGlRenderState s_cam(
	pangolin::ProjectionMatrix(1600,1000,680,680,800,500,0.2,100),
	pangolin::ModelViewLookAt(0,4,2, 0,0,0, pangolin::AxisY)
	);

	// Object to hold all scene objects
	pangolin::Renderable scene;

	// Create Interactive View in window
	pangolin::SceneHandler handler(scene,s_cam);
	pangolin::View& d_cam = pangolin::CreateDisplay()
	    .SetBounds(0.0, 1.0, 0.0, 1.0, -1600.0f/1000.0f)
	    .SetHandler(&handler);

	d_cam.SetDrawFunction([&](pangolin::View& view){
	view.Activate(s_cam);
	scene.Render();
	});
	
	// Open up video stream
	cv::VideoCapture vid1(INPUT_vid_location);

	// Safety check
	if (!vid1.isOpened())
	{
		throw std::invalid_argument("Invalid input video stream");
		return -1;
	}

	// Init camera/sensor config
	// Grab one frame to check input image dims
	cv::Mat initImg;
	vid1 >> initImg;
	double img_scale = 1;
	if (initImg.cols > 960) img_scale = initImg.cols/960;
	CameraParams<double> camera(
			img_scale, initImg.cols, initImg.rows, INPUT_f
			);

	// Store 2 most recent frames for feature matching
	std::queue<cv::Mat> vid_frames;
	bool isVidBuff = true;

	// Main program loop
	while( !pangolin::ShouldQuit() )
	{
		// Clear screen and activate view to render into
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		if (isVidBuff)
		{
			cv::Mat video_frame;
			int extra_skip = 0;

			// Capture frame
			vid1 >> video_frame;

			// If SKIP > 0 skips extra frames 
			while( !video_frame.empty() && extra_skip < INPUT_skip)
			{
				vid1 >> video_frame;
				extra_skip++;
			}

			// If video stream is finished stop reading it but allow for OpenGl loop
			if ( video_frame.empty() )
			{
				isVidBuff = false;
				continue;
			}

			// Push frame to the vector for use in "doFrame" function
			vid_frames.push(video_frame);

		}

		// All the action happens inside
		doFrame(scene, camera, vid_frames);	

		// Swap frames and Process Events
		pangolin::FinishFrame();

		// remove the previous frame
		if (isVidBuff && vid_frames.size() > 1) vid_frames.pop();
	}

	vid1.release();
	cv::waitKey(0);

return 0;
}
