# Pop up from Single Image #
This code contains algorithm to pop up a 3D plane model from images. Given a RGB and ground segmentation image, the algorithm detects edges and selecs the ground-wall boundary edges then pops up 3D point cloud.

**Authors:** [Shichao Yang](http://www.frc.ri.cmu.edu/~syang/), [Sebastian Scherer](http://theairlab.org/)

**Related Paper:**

* **Real-time 3D Scene Layout from a Single Image Using Convolutional Neural Networks**, ICRA 2017, S. Yang, D. Maturana, S. Scherer  [**PDF**](http://www.frc.ri.cmu.edu/~syang/Publications/icra_2016.pdf)
* **Pop-up SLAM: Semantic Monocular Plane SLAM for Low-texture Environments**, IROS 2017, S. Yang, Y. Song, M. Kaess, S. Scherer [**PDF**](http://www.frc.ri.cmu.edu/~syang/Publications/iros_2016.pdf)

If you use the code in your research work, please cite the above paper. Please do not hesitate to contact the authors if you have any further questions.



# Installation#

### Prerequisites
This code contains several ros packages. We test it in **ROS indigo + Ubuntu 14.04**. Create or use existing a ros workspace.
```bash
mkdir -p ~/popup_ws/src
cd ~/popup_ws/src
catkin_init_workspace
git clone TODO改掉！！！
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

