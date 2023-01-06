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
	pangolin::ModelViewLookAt(0,3,5, 0,0,0, pangolin::AxisY)
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
	
	// Open up video stream
	cv::VideoCapture vid1("./data/vid/cem_3.mp4");

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
	bool isVidBuff = true;

	// Skip frames every buffer read, _SKIP = 0 means no frame skipping
	int _SKIP =6;

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

			// If _SKIP > 0 skips extra frames 
			while( !video_frame.empty() && extra_skip < _SKIP)
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
