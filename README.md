# Monocular Visual Odometry

Camera pose estimation based on the relative movement found using keypoints between two succeding frames.
Every new frame new keypoints are found and matched with appropriate keypoints from previous frame. Then, Essential Matrix is estimated(using OpenCV's Nister's 5-point algorithm implementation) and a pose is recovered from that, which is a rotation and translation matrix between camera 1/previous frame and camera 2/current frame.

All computer vision tasks has been implemented via OpenCV library, rendering done via Pangolin.

Full example video can be found on [youtube](https://www.youtube.com/watch?v=5iN7dfLXRhU)

![Cemetery example](https://github.com/tomgasper/vis-odo/blob/main/data/preview.gif?raw=true)

# Usage
Instrinsic camera parameteres(namely focal length) have to be known and provided by the user via command line arguments. Focal length must be in pixels and must be scaled as to match 960 pixels width image(frames are being resized for the processing).

So for example if you input 1920x1080 camera recording and the focal length of that video is 1327 then you should resize it as it was a focal length for 960x540 - 1327/(1920/960) = 663.5

After compilation start the program from command line using exactly 3 arguments
```
./main <focal length> <frames to skip> <video directory>
```

Example (using POCO X3 PRO footage)
```
./main 663 4 ./data/cem.mp4
```

# Notes

This is a very simple visual odometry implementation done for learning purposes. The generated camera trajectory is generally not accurate. There's no proper camera instatiation and the program doesn't work correctly when there's just a rotation movement without any pose translation(limitations of monocular visual odometry)

## Dependencies
* C++11 
* Pangolin - 3d scene and display
* Eigen - matrix operations
* OpenCV 4.6 - 2d features, essential matrix and triangulation
