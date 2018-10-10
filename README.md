# Pop Single Image #
This code contains algorithm to pop up a 3D plane model from images. Given a RGB and ground segmentation image, the algorithm detects edges and selecs the ground-wall boundary edges then pops up 3D point cloud. It is extended to pop-up [plane slam](https://github.com/shichaoy/pop_up_slam).

**Authors:** [Shichao Yang](http://www.frc.ri.cmu.edu/~syang/)

**Related Paper:**

* **Real-time 3D Scene Layout from a Single Image Using Convolutional Neural Networks**, ICRA 2016, S. Yang, D. Maturana, S. Scherer  [**PDF**](http://www.frc.ri.cmu.edu/~syang/Publications/icra_2016_sinpop.pdf)
* **Pop-up SLAM: Semantic Monocular Plane SLAM for Low-texture Environments**, IROS 2016, S. Yang, Y. Song, M. Kaess, S. Scherer [**PDF**](http://www.frc.ri.cmu.edu/~syang/Publications/iros_2016_popslam.pdf)

If you use the code in your research work, please cite the above paper. Please do not hesitate to contact the authors if you have any further questions.



# Installation

### Prerequisites
This code contains several ros packages. We test it in **ROS indigo/kinetic, Ubuntu 14.04/16.04, Opencv 2/3**. Create or use existing a ros workspace.
```bash
mkdir -p ~/popup_ws/src
cd ~/popup_ws/src
catkin_init_workspace
git clone git@github.com:shichaoy/pop_up_image.git
cd pop_up_image
```

### Install dependency packages of python
```bash
sh install_dependenices.sh
```

### Compile
```bash
cd ~/popup_ws
catkin_make
```


# Running #
```bash
source devel/setup.bash
roslaunch pop_up_wall pop_main_sample.launch
```
You will see point cloud in Rviz. Change the image id in the launch file to test more examples stored under pop_up_wall/data.

### Notes

1. If it shows "NameError: 'pop_up fun...params' is not defined". That is due to python dependency modules are not installed properly. Make sure "from skimage.measure import find_contours,approximate_polygon" can work alone in python. Also 'souce setup.bash' when python/pop_up_python changes/recompiles.  There is some pop-up image python function I cannot find C++ replacement therefore we have to use python... The main part is in C++.
