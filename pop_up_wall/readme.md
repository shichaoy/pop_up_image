# Description #
This package contains to pop up 3D plane model. Given segmented image (ground vs wall), this package is to fit or select edges then pop up to 3D point cloud.

**Authors:** Shichao Yang (shichaoy@andrew.cmu.edu)

**Related Paper**

* **Real-time 3D Scene Layout from a Single Image Using Convolutional Neural Networks**, ICRA 2017, S. Yang, D. Maturana, S. Scherer
* **Pop-up SLAM: Semantic Monocular Plane SLAM for Low-texture Environments**, IROS 2017, S. Yang, Y. Song, M. Kaess, S. Scherer

# How to run.
1. catkin_make.
2. `roslaunch pop_up_wall pop_main.launch`  in the launch file set the image id. Images are stored under data/  The main file contains more parameters to set



