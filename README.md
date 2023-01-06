# Monocular Visual Odometry

Camera pose estimation based on the relative movement found using keypoints between two succeding frames.

Every new frame new keypoints are found and matched with appropriate keypoints from previous frame. Then, Essential Matrix is estimated(using OpenCV's Nister's 5-point algorithm implementation) and a pose is recovered from that, which is a rotation and translation matrix between camera 1/previous frame and camera 2/current frame.

All the computer vision tasks have been implemented via OpenCV library, rendering done via Pangolin.

Full example video can be found on [youtube](https://www.youtube.com/watch?v=5iN7dfLXRhU)

![Cemetery example](https://github.com/tomgasper/vis-odo/blob/main/data/preview.gif?raw=true)

# Usage
Instrinsic camera parameteres(namely focal length) have to be known and provided by the user via command line arguments.

```
./main <focal length> <frames to skip> <video directory>
```

# Notes

This is a very simple visual odometry implementation done for learning purposes. The generated camera trajectory is generally not accurate. There's no proper camera instatiation and the program doesn't work correctly when there's just a rotation movement without any pose translation(limitations of monocular visual odometry)

## Dependencies
* C++11 
* Pangolin - 3d scene and display
* Eigen - matrix operations
* OpenCV 3.4.14 - 2d features, essential matrix and triangulation
