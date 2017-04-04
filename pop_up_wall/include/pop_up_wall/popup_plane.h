/*
 * Copyright Shichao Yang,2016, Carnegie Mellon University
 * Email: shichaoy@andrew.cmu.edu 
 */
// std c
#pragma once

#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <ctime>
#include <assert.h>     /* assert */
#include <tuple>
// opencv pcl
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/passthrough.h>

// ros
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <image_geometry/pinhole_camera_model.h>
#include <pcl_conversions/pcl_conversions.h>

// Eigen
#include <Eigen/Dense>
#include <Eigen/LU>

// boost python to call python pop up
#include <boost/unordered_map.hpp>
#include <boost/python.hpp>
#include <python2.7/Python.h>


// ours
#include "pop_up_wall/matrix_utils.h"
#include "line_lbd/line_lbd_allclass.h"

using namespace std;
using namespace cv;
using namespace Eigen;
namespace bp = boost::python;


typedef pcl::PointCloud<pcl::PointXYZRGB> pclRGB;


class popup_plane
{
protected:  
  int width;
  int height;
  Matrix3f Kalib;
  Matrix3f invK;  
 
  bp::object* py_module;
  bp::object py_dictionary;
  bool inited_python;
public:
  ros::NodeHandle nn;  // to parse the parameters

  popup_plane();
  ~popup_plane();

  cv::Mat edge_label_img;  // ground region formed by edges    ground is 255, wall is 0, void is 127  

  void Init_python();	//using python function to get ground polygons, generatly take ~5 ms more 

  line_lbd_detect* line_lbd_ptr;
  
//==============================================   fixed 2d information after getting ground edges.  ===========================  
  int total_plane_number; // // include the ground  0, or 2,3,... at least a ground a a wall
  MatrixXf ground_seg2d_lines_connect;  // n*4 close polygons edges  contain some manually connected edges
  MatrixXf ground_seg2d_lines_open; // open polygons edges
  
  
//==============================================   3d information related to pose.  ==============================================    
//all the closed polygons include ground, manually connected edges, each is 2*n image coordinate  used by slam to do association  doesn't consider ceilling height
  vector<MatrixXf> all_closed_2d_bound_polygons;  // pose will affect this

  MatrixXf all_planes_world;   // store all the plane parameters include ground in world frame, each row is a plane  n*4  inclue manually connected part
  MatrixXf all_planes_sensor;  // store all the plane parameters include ground in sensor frame, each row is a plane n*4   inclue manually connected part   
  Vector4f ceiling_plane_world;
  Vector4f ceiling_plane_sensor;

  int good_plane_number; //// include the ground
  VectorXf good_plane_indices; // ground and good wall indices in all the planes  manually connected plane  
  MatrixXf ground_seg3d_lines_world;  // 3D world coordinates of ground polygons. each row is two points x1 y1 x2 y2
  MatrixXf ground_seg3d_lines_sensor; //  sensor coordinate of ground polygons  . each row is two points x1 y1 z1 x2 y2 z2  
  
  vector<MatrixXf> all_3d_bound_polygons_world;  //3D closed bounding polygon of each ground and wall, each is 3*n world coordinate  considering the ceiling height  
  VectorXf all_plane_dist_to_cam;  // wall plane distance to the camera  include the ground, and manually edges
  pclRGB::Ptr pcl_cloud_world;
  std::vector<pclRGB::Ptr> partplane_clouds_open_all;  
  
  
 //==============================================    interface functions  ======================================================
  void set_calibration(Matrix3f calib);
    
  // //given the rgb (or gray) image, label image(0-255), get the 2D ground edge, no need of pose,  ground is labeled 255 (not 1), wall label is 0. must be uint8.  
  void get_ground_edges(const cv::Mat &raw_img, const cv::Mat &label_map);

  // from the ground edges and current pose, compute the plane equation, also call find_2d_3d_closed_polygon to compute polygons  can affect good plane indice
  // transToWolrd a camera space point multiplied by this, goes to world frame.
  void get_plane_equation(const Matrix4f& transToWolrd, const MatrixXf& ground_seg2d_lines_connect_cp);
  
  //create 3d point cloud in world frame 3*n (pcl_cloud ptr) wall plane indices (from 0, not including ground).  old name(python name)  pop_up_3dcloud_closed_polygons_transforms; 
  void generate_cloud(const cv::Mat &bgr_img,const Matrix4f& transToWolrd, bool downsample_poly=false, bool separate_cloud=false,
		      bool mixed_cloud=true,bool get_label_img=false,float depth_thre=10.0);
    
  
// ============================================== -utility function   ============================================== 
  // matrix to cloud
  void matrixToCloud(const vector<MatrixXf> &local_pts_parts, const vector<MatrixXf> &global_pts_parts, const vector<MatrixX8uu> &rgb_mat_parts,
		     bool separate_cloud=false,bool mixed_cloud=true,float depth_thre=10.0);  
  
  //given the ground 2d polygon segments n*2, find and update all the wall plane's close polygons
  void find_2d_3d_closed_polygon(const Matrix4f& transToWolrd);
      
  // given raw edges, then only call python to find contours, interval tree, then most other parts are written in C++
  tuple<MatrixXf, MatrixXf, VectorXf> edge_get_polygons(cv::Mat &lsd_edges,const cv::Mat &label_map); // python get contour
  
  // give a point pt in the image with (0,0) origin, with a direction in the image (delta_u,delta_v). 
  // find the hippting point of top or left/right boundary.
  Vector2f direction_hit_boundary(Vector2f pt,Vector2f direc); 

  //input is closed polygons, n*2, each row is a points, output is 3*m homo coord, and 3*m rgb matrix. all the parameters provided. don't need class variables anymore.
  void closed_polygons_homo_pts(const MatrixXf &polys_close,MatrixXf &inside_pts_homo_out, bool downsample);  
  // also retrieve the rgb color of each pixels.
  void closed_polygons_homo_pts(const MatrixXf &polys_close,MatrixXf &inside_pts_homo_out,const cv::Mat &bgr_img, MatrixX8uu &part_polygon_rgb,
				bool downsample);
  
  
// image process parameters
  int erosion_distance;
  int dilation_distance;
  
// python pop up parameters
  double pre_vertical_thre=15;
  double pre_minium_len=15;
  double pre_contour_close_thre=50;
  double interval_overlap_thre=20;
  double post_short_thre=30;
  double post_bind_dist_thre=10;
  double post_merge_dist_thre=20;
  double post_merge_angle_thre=10;
  double post_extend_thre=15;
  
  double pre_boundary_thre=5;
  double pre_merge_angle_thre=10;
  double pre_merge_dist_thre=10; 
  double pre_proj_angle_thre=20; 
  double pre_proj_cover_thre=0.6;
  double pre_proj_cover_large_thre=0.8;
  double pre_proj_dist_thre=100;	    
	    
  double ceiling_threshold; // ceiling height threshold
  int cloud_downsample_rate=20;  // only used in final fetching. actually matter less
  bool downsample_contour=false;
  bool use_fast_edges_detection=false;
  // 3d plane distance threshold
  double plane_cam_dist_thre=10;
  
  
};