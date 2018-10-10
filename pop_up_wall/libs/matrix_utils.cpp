/*
 * Copyright Shichao Yang,2016, Carnegie Mellon University
 * Email: shichaoy@andrew.cmu.edu
 *
 */

#include "pop_up_wall/matrix_utils.h"

//output n*2
void points_to_matrix(const vector<Point> &cv_pts_in,MatrixXf &eigen_pts_out)
{
   eigen_pts_out.resize(cv_pts_in.size(),2);
   for (int ind=0;ind<cv_pts_in.size();ind++)
   {
     eigen_pts_out(ind,0)=cv_pts_in[ind].x;
     eigen_pts_out(ind,1)=cv_pts_in[ind].y;
   }
}

// n*3
void points_to_homo_matrix(const vector<Point> &cv_pts_in,MatrixXf &eigen_homo_out)
{
   eigen_homo_out.resize(cv_pts_in.size(),3);
   for (int ind=0;ind<cv_pts_in.size();ind++)
   {
     eigen_homo_out(ind,0)=cv_pts_in[ind].x;
     eigen_homo_out(ind,1)=cv_pts_in[ind].y;
     eigen_homo_out(ind,2)=1;
   }   
}

//input  n*2
void matrix_to_points(const MatrixXf &eigen_pts_int,vector<Point> &cv_pts_out)
{
   cv_pts_out.resize(eigen_pts_int.rows());   
   for (int ind=0;ind<eigen_pts_int.rows();ind++)
     cv_pts_out[ind]=Point(eigen_pts_int(ind,0),eigen_pts_int(ind,1));    
}

//vertical vector transpose it.
void vert_stack_v(const MatrixXf &a_in, const VectorXf &b_in, MatrixXf &combined_out)
{
   assert (a_in.cols() == b_in.rows());
   combined_out.resize(a_in.rows()+1,a_in.cols());
   combined_out<<a_in,
	         b_in.transpose();   
}

void vert_stack_v_self(MatrixXf &a_in, const VectorXf &b_in)
{
   assert (a_in.cols() == b_in.rows());
   MatrixXf combined_out(a_in.rows()+1,a_in.cols());
   combined_out<<a_in,
		 b_in.transpose();
   a_in=combined_out;
}

void vert_stack_m(const MatrixXf &a_in, const MatrixXf &b_in, MatrixXf &combined_out)
{
   assert (a_in.cols() == b_in.cols());
   combined_out.resize(a_in.rows()+b_in.rows(),a_in.cols());
   combined_out<<a_in,
		 b_in;
}
void vert_stack_m_self(MatrixXf &a_in, const MatrixXf &b_in)
{
   assert (a_in.cols() == b_in.cols());
   MatrixXf combined_out(a_in.rows()+b_in.rows(),a_in.cols());
   combined_out<<a_in,
		 b_in;
    a_in=combined_out;
}
void hori_stack_m_self(MatrixXf &a_in, const MatrixXf &b_in)
{
   assert (a_in.rows() == b_in.rows());
   MatrixXf combined_out(a_in.rows(),a_in.cols()+b_in.cols());
   combined_out<<a_in, b_in;
   a_in=combined_out;
}
void hori_stack_v_self(MatrixXf &a_in, const VectorXf &b_in)
{
   assert (a_in.rows() == b_in.rows());
   MatrixXf combined_out(a_in.rows(),a_in.cols()+1);
   combined_out<<a_in, b_in;
   a_in=combined_out;
}

void hori_stack_vec_m(const vector<MatrixXf> &mat_vec_in, MatrixXf &combined_out)
{
 //  assert (a_in.rows() == b_in.rows());
  int total_column=0;
  for (int i=0;i<mat_vec_in.size();i++)
    total_column+=mat_vec_in[i].cols();
  
  combined_out.resize(mat_vec_in[0].rows(),total_column);  // TODO hack here, I store only 10 walls   this is faster than push back one by one
  combined_out<<mat_vec_in[0], mat_vec_in[1],mat_vec_in[2],mat_vec_in[3],mat_vec_in[4],mat_vec_in[5], \
		mat_vec_in[6],mat_vec_in[7],mat_vec_in[8],mat_vec_in[9],mat_vec_in[10],mat_vec_in[11],
		mat_vec_in[12],mat_vec_in[13],mat_vec_in[14];
}


void hori_stack_vec_m(const vector<MatrixX8uu> &mat_vec_in, MatrixX8uu &combined_out)
{
 //  assert (a_in.rows() == b_in.rows());
  int total_column=0;
  for (int i=0;i<mat_vec_in.size();i++)
    total_column+=mat_vec_in[i].cols();
  
  combined_out.resize(mat_vec_in[0].rows(),total_column);  // TODO hack here, I store only 10 walls   this is faster than push back one by one
  combined_out<<mat_vec_in[0], mat_vec_in[1],mat_vec_in[2],mat_vec_in[3],mat_vec_in[4],mat_vec_in[5], \
		mat_vec_in[6],mat_vec_in[7],mat_vec_in[8],mat_vec_in[9],mat_vec_in[10],mat_vec_in[11],
		mat_vec_in[12],mat_vec_in[13],mat_vec_in[14];
}


//rays is 3*n, each column is a ray staring from origin  plane is (4，1） parameters, compute intersection  output is 3*n 
void ray_plane_interact(const MatrixXf &rays,const Eigen::Vector4f &plane,MatrixXf &intersections)
{  
  VectorXf frac=-plane[3]/(plane.head(3).transpose()*rays).array();   //n*1 
  intersections= frac.transpose().replicate<3,1>().array() * rays.array();
}


//rays is 3*n, each column is a ray staring from origin  plane is (4，1） parameters, compute intersection  output is 3*n, another way
void ray_plane_interact_mod(const MatrixXf &rays,const Eigen::Vector4f &plane,MatrixXf &intersections)
{ 
  MatrixXf H=Matrix<float, 4, 3>::Identity();
  H.row(3)=-plane.head(3)/plane(3);
  homo_to_real_coord(H*rays,intersections);  
}


// find the intersection of two point line segments with a z plane of height
bool points_intersect_plane(const Vector3f &pt_a,const Vector3f &pt_b, float height,Vector3f &intersect)
{
  float lambda=(height-pt_a(2))/(pt_b(2)-pt_a(2));
  if (lambda>=0 && lambda<=1)
  {
    intersect=pt_a+lambda*(pt_b-pt_a);
    return true;
  }
  return false;
}

float point_dist_lineseg(const Vector2f &begin_pt, const Vector2f &end_pt, const Vector2f &query_pt)
{  // v w p
    // Return minimum distance between line segment vw and point p
  float length = (end_pt-begin_pt).norm();
  if (length < 0.001) return (query_pt-begin_pt).norm();   // v == w case
  // Consider the line extending the segment, parameterized as v + t (w - v).
  // We find projection of point p onto the line. 
  // It falls where t = [(p-v) . (w-v)] / |w-v|^2
  float t = ((query_pt-begin_pt).dot(end_pt-begin_pt))/length/length;
  if (t < 0.0) return (query_pt-begin_pt).norm();       // Beyond the 'v' end of the segment
  else if (t > 1.0) return (query_pt-end_pt).norm();  // Beyond the 'w' end of the segment
  const Vector2f projection = begin_pt + t * (end_pt - begin_pt);  // Projection falls on the segment
  return (query_pt-projection).norm();
}

float point_proj_lineseg(const Vector2f &begin_pt, const Vector2f &end_pt, const Vector2f &query_pt)
{  // v w p
    // Return minimum distance between line segment vw and point p
  float length = (end_pt-begin_pt).norm();
  if (length < 0.001) return (query_pt-begin_pt).norm();   // v == w case
  // Consider the line extending the segment, parameterized as v + t (w - v).
  // We find projection of point p onto the line. 
  // It falls where t = [(p-v) . (w-v)] / |w-v|^2
  float t = ((query_pt-begin_pt).dot(end_pt-begin_pt))/length/length;
  return t;  
}


float point_dist_line(const Vector2f &begin_pt, const Vector2f &end_pt, const Vector2f &query_pt)
{  // v w p
    // Return minimum distance between line segment vw and point p
  float length = (end_pt-begin_pt).norm();
  if (length < 0.001) return (query_pt-begin_pt).norm();   // v == w case
  // Consider the line extending the segment, parameterized as v + t (w - v).
  // We find projection of point p onto the line. 
  // It falls where t = [(p-v) . (w-v)] / |w-v|^2
  float t = ((query_pt-begin_pt).dot(end_pt-begin_pt))/length/length;
  const Vector2f projection = begin_pt + t * (end_pt - begin_pt);  // Projection falls on the segment
  return (query_pt-projection).norm();
}

void point_distproj_to_line(const Vector2f &begin_pt, const Vector2f &end_pt, const Vector2f &query_pt, float& dist, float& proj_percent)
{
      // Return minimum distance between line segment vw and point p
    float length = (end_pt-begin_pt).norm();
    if (length < 0.001){  // very short lines segments
	dist=(query_pt-begin_pt).norm();   // v == w case
	proj_percent=-1;
	return;
    }
    // Consider the line extending the segment, parameterized as v + t (w - v).
    // We find projection of point p onto the line. 
    // It falls where t = [(p-v) . (w-v)] / |w-v|^2
    float t = ((query_pt-begin_pt).dot(end_pt-begin_pt))/length/length;
    const Vector2f projection = begin_pt + t * (end_pt - begin_pt);  // Projection falls on the segment
    dist=(query_pt-projection).norm();
    if (t>1)  // cut into [0 1]
      t=1;
    if (t<0)
      t=0;    
    proj_percent = t;
    return;
}


// input is 3*n (2*n)  output is 4*n(3*n)
void real_to_homo_coord(const MatrixXf &pts_in,MatrixXf &pts_homo_out)
{
  int raw_rows=pts_in.rows();
  int raw_cols=pts_in.cols();
  
  pts_homo_out.resize(raw_rows+1,raw_cols);
  pts_homo_out<<pts_in,
	        RowVectorXf::Ones(raw_cols);
}

// input is 3*n (2*n)  output is 4*n(3*n)
MatrixXf real_to_homo_coord(const MatrixXf &pts_in)
{
  MatrixXf pts_homo_out;
  int raw_rows=pts_in.rows();
  int raw_cols=pts_in.cols();
  
  pts_homo_out.resize(raw_rows+1,raw_cols);
  pts_homo_out<<pts_in,
	        RowVectorXf::Ones(raw_cols);
  return pts_homo_out;
}

VectorXf real_to_homo_coord_vec(const VectorXf &pt_in)
{
  VectorXf pt_homo_out;
  int raw_rows=pt_in.rows();  
  
  pt_homo_out.resize(raw_rows+1);
  pt_homo_out<<pt_in,
	       1;
  return pt_homo_out;
}


// input is 3*n(4*n)  output is 2*n(3*n)
void homo_to_real_coord(const MatrixXf &pts_homo_in, MatrixXf &pts_out)
{  
  if (pts_homo_in.rows()==4)
      pts_out=pts_homo_in.topRows(3).array()/pts_homo_in.bottomRows(1).replicate<3,1>().array();   //replicate needs actual number, cannot be M or N
  else if (pts_homo_in.rows()==3)
      pts_out=pts_homo_in.topRows(2).array()/pts_homo_in.bottomRows(1).replicate<2,1>().array(); 
}
MatrixXf homo_to_real_coord(const MatrixXf &pts_homo_in)
{  
  MatrixXf pts_out;
  if (pts_homo_in.rows()==4)
      pts_out=pts_homo_in.topRows(3).array()/pts_homo_in.bottomRows(1).replicate<3,1>().array();   //replicate needs actual number
  else if (pts_homo_in.rows()==3)
      pts_out=pts_homo_in.topRows(2).array()/pts_homo_in.bottomRows(1).replicate<2,1>().array(); 
  return pts_out;
}

VectorXf homo_to_real_coord_vec(const VectorXf &pt_homo_in)
{
  VectorXf pt_out;
  if (pt_homo_in.rows()==4)
    pt_out=pt_homo_in.head(3)/pt_homo_in(3);
  else if (pt_homo_in.rows()==3)
    pt_out=pt_homo_in.head(2)/pt_homo_in(2);

  return pt_out;
}


// ground in the label map is 1 or 255 (not 0)
void blend_segmentation(Mat& raw_color_img, Mat& label_map, Mat& blend_img)
{
  blend_img=raw_color_img.clone();  
  cv::Vec3b bgr_value;
  
  vector<Point2i> ground_pts;
  findNonZero(label_map, ground_pts);    
  for (int ind=0;ind<ground_pts.size();ind++)
  {
      bgr_value=blend_img.at<cv::Vec3b>(ground_pts[ind].y,ground_pts[ind].x);  //at row (y) and col(x)      
      bgr_value=bgr_value*0.8+cv::Vec3b(0,255,0)*0.2;  // uint to float???
      for (int i=0;i<3;i++)
      {
	if (bgr_value(i)>255)
	  bgr_value(i)=255;
      }	
      blend_img.at<cv::Vec3b>(ground_pts[ind].y,ground_pts[ind].x)=bgr_value;
  }  
}


bool check_element_in_vector(const float& element, const VectorXf &vec_check)
{
  for (int i=0;i<vec_check.rows();i++)  
    if (element==vec_check(i))    
	return true;
  return false;
}

bool check_element_in_vector(const int& element, const VectorXi &vec_check)
{
  for (int i=0;i<vec_check.rows();i++)
    if (element==vec_check(i))
	return true;
  return false;
}


//change angle from [-180,180] to [-90,90]
float normalize_to_pi(float angle)
{
    if (angle > 90)
        return angle-180; // # change to -90 ~90
    else if (angle<-90)
        return angle+180;
    else
        return angle;
}

