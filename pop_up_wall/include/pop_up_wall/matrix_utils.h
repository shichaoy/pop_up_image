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

// opencv pcl
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

// Eigen
#include <Eigen/Dense>
#include <Eigen/LU>


using namespace std;
using namespace cv;
using namespace Eigen;

typedef Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> MatrixX8uu;

//output n*2    opencv point vector to eigen matrix
void points_to_matrix(const vector<Point> &cv_pts_in,MatrixXf &eigen_pts_out);
//output n*3    opencv point vector to eigen matrix homogeneous coordinates
void points_to_homo_matrix(const vector<Point> &cv_pts_in,MatrixXf &eigen_homo_out);
//input  n*2    eigen matrix to opencv point vector
void matrix_to_points(const MatrixXf &eigen_pts_int,vector<Point> &cv_pts_out);

//vertically stack a vertical vector to a matrix
void vert_stack_v(const MatrixXf &a_in, const VectorXf &b_in, MatrixXf &combined_out);
//vertically stack a vertical vector to a matrix itself, not creating new
void vert_stack_v_self(MatrixXf &a_in, const VectorXf &b_in);
//vertically stack a matrix to a matrix
void vert_stack_m(const MatrixXf &a_in, const MatrixXf &b_in, MatrixXf &combined_out);
//vertically stack a matrix to a matrix itself, not creating new
void vert_stack_m_self(MatrixXf &a_in, const MatrixXf &b_in);
//horizontal stack a matrix to a matrix itself, not creating new
void hori_stack_m_self(MatrixXf &a_in, const MatrixXf &b_in);
//horizontal stack a column vector to a matrix itself, not creating new
void hori_stack_v_self(MatrixXf &a_in, const VectorXf &b_in);
//horizontal stack a series of matrix
void hori_stack_vec_m(const vector<MatrixXf> &mat_vec_in, MatrixXf &combined_out);
void hori_stack_vec_m(const vector<MatrixX8uu> &mat_vec_in, MatrixX8uu &combined_out);

//rays is 3*n, each column is a ray starting from origin
//plane is (4，1） parameters, compute intersection    output is 3*n
void ray_plane_interact(const MatrixXf &rays,const Eigen::Vector4f &plane,MatrixXf &intersections);
void ray_plane_interact_mod(const MatrixXf &rays,const Eigen::Vector4f &plane,MatrixXf &intersections);

// find the intersection of two point line segments with a z plane of height
bool points_intersect_plane(const Vector3f &pt_a,const Vector3f &pt_b, float height,Vector3f &intersect);

// comptue point cloest distance to a line segments
float point_dist_lineseg(const Vector2f &begin_pt, const Vector2f &end_pt, const Vector2f &query_pt);

// comptue point cloest distance to a infinite line
float point_dist_line(const Vector2f &begin_pt, const Vector2f &end_pt, const Vector2f &query_pt);


// query point distance and projection to a infinite line defined by two points
void point_distproj_to_line(const Vector2f &begin_pt, const Vector2f &end_pt, const Vector2f &query_pt, float& dist, float& proj_percent);

// input is 3*n (or 2*n)  output is 4*n (or 3*n)
void real_to_homo_coord(const MatrixXf &pts_in,MatrixXf &pts_homo_out);

// input is 3*n (or 2*n)  output is 4*n (or 3*n)
MatrixXf real_to_homo_coord(const MatrixXf &pts_in);
VectorXf real_to_homo_coord_vec(const VectorXf &pts_in);


// input is 3*n (or 4*n)  output is 2*n(or 3*n)
void homo_to_real_coord(const MatrixXf &pts_homo_in, MatrixXf &pts_out);

// input is 3*n (or 4*n)  output is 2*n(or 3*n)
MatrixXf homo_to_real_coord(const MatrixXf &pts_homo_in);
VectorXf homo_to_real_coord_vec(const VectorXf &pt_homo_in);

// ground in the label map is 1 or 255 (not 0)
void blend_segmentation(Mat& raw_color_img, Mat& label_map, Mat& blend_img);

bool check_element_in_vector(const float& element, const VectorXf &vec_check);
bool check_element_in_vector(const int& element, const VectorXi &vec_check);

float normalize_to_pi(float angle);