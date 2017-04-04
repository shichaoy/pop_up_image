/*
 * line_detection interface
 * Copyright Shichao Yang,2016, Carnegie Mellon University
 * Email: shichaoy@andrew.cmu.edu
 *
 */

#include "pop_up_wall/popup_plane.h"
#include "pop_up_wall/matrix_utils.h"
// #include "interval_tree/IntervalTree.h"

using namespace std;
using namespace cv;
using namespace Eigen;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_row;

popup_plane::popup_plane():nn( "~" )
{ 
    width=640;
    height=480;
  //   Kalib << 570, 0, 320,  // kinect
  // 	   0, 570, 240, 
  // 	   0,   0,   1;
    Kalib << 371.975264, 0, 315.632343,   //ueye
	    0, 372.163582, 250.592551, 
	    0,   0,   1;	   
    invK=Kalib.inverse();
    
    all_planes_world.resize(1,4);
    all_planes_sensor.resize(1,4);
      
    erosion_distance=11;
    dilation_distance=11;
    
    ceiling_threshold=2.5;
    pcl_cloud_world.reset(new pclRGB);
    
    inited_python=false;

    line_lbd_ptr = new line_lbd_detect();
    
  // receive the paramters for python pop up  
    nn.param ("/pre_vertical_thre", pre_vertical_thre, pre_vertical_thre);
    nn.param ("/pre_minium_len", pre_minium_len, pre_minium_len);
    nn.param ("/pre_contour_close_thre", pre_contour_close_thre, pre_contour_close_thre);
    nn.param ("/interval_overlap_thre", interval_overlap_thre, interval_overlap_thre);
    nn.param ("/post_short_thre", post_short_thre, post_short_thre);
    nn.param ("/post_bind_dist_thre", post_bind_dist_thre, post_bind_dist_thre);
    nn.param ("/post_merge_dist_thre", post_merge_dist_thre, post_merge_dist_thre);
    nn.param ("/post_merge_angle_thre", post_merge_angle_thre, post_merge_angle_thre);
    nn.param ("/post_extend_thre", post_extend_thre, post_extend_thre);
        
    nn.param ("/downsample_contour", downsample_contour, downsample_contour);
    nn.param ("/use_fast_edges_detection", use_fast_edges_detection, use_fast_edges_detection);
    nn.param ("/cloud_downsample_rate", cloud_downsample_rate, cloud_downsample_rate);
    nn.param ("/plane_cam_dist_thre", plane_cam_dist_thre, plane_cam_dist_thre);  
}

popup_plane::~popup_plane()
{
    if (inited_python)
    {
      delete py_module;
      Py_Finalize();
    }
}

void popup_plane::set_calibration(Matrix3f calib)
{
    Kalib=calib;
    invK=Kalib.inverse();  
}
  

void popup_plane::closed_polygons_homo_pts(const MatrixXf &polys_close,MatrixXf &inside_pts_homo_out, bool downsample_poly)
{
    MatrixXf new_polys_close=polys_close;
    if (downsample_poly)
	new_polys_close=new_polys_close/2;
    
    vector<Point> polygon;
    matrix_to_points(new_polys_close,polygon);  
    Rect bounding_box=boundingRect(polygon);
    MatrixXf polys_close_shift=new_polys_close;  
    polys_close_shift.rowwise() -= Vector2f(bounding_box.x,bounding_box.y).transpose();
    cv::Mat raw_img_cp8(Size(bounding_box.width, bounding_box.height),CV_8UC1,Scalar(0));  
    vector<Point> polygon_shift;
    matrix_to_points(polys_close_shift,polygon_shift);
    fillConvexPoly(raw_img_cp8,&polygon_shift[0], new_polys_close.rows(), Scalar(255,0,0), 0,0);    //faster than point in polygon test.
    vector<Point2i> locations_shift;   // output, locations of non-zero pixels 
    findNonZero(raw_img_cp8, locations_shift);  
    
    inside_pts_homo_out.resize(3,locations_shift.size());
    for (int ind=0;ind<locations_shift.size();ind++)
    {
	if (downsample_poly){
	    inside_pts_homo_out(0,ind)=(locations_shift[ind].x+bounding_box.x)*2;
	    inside_pts_homo_out(1,ind)=(locations_shift[ind].y+bounding_box.y)*2;
	    inside_pts_homo_out(2,ind)=1;
	}
	else{
	    inside_pts_homo_out(0,ind)=locations_shift[ind].x+bounding_box.x;
	    inside_pts_homo_out(1,ind)=locations_shift[ind].y+bounding_box.y;
	    inside_pts_homo_out(2,ind)=1;  
	}
    }
  //   inside_pts_homo_out.colwise() += Vector3f(bounding_box.x,bounding_box.y,0); 
}

void popup_plane::closed_polygons_homo_pts(const MatrixXf &polys_close,MatrixXf &inside_pts_homo_out,const cv::Mat &bgr_img, MatrixX8uu &part_polygon_bgr, bool downsample_poly)
{
    MatrixXf new_polys_close=polys_close;
    if (downsample_poly)
	new_polys_close=new_polys_close/2;
    
    vector<Point> polygon;
    matrix_to_points(new_polys_close,polygon);  
    Rect bounding_box=boundingRect(polygon);
    MatrixXf polys_close_shift=new_polys_close;  
    polys_close_shift.rowwise() -= Vector2f(bounding_box.x,bounding_box.y).transpose();
    cv::Mat raw_img_cp8(Size(bounding_box.width, bounding_box.height),CV_8UC1,Scalar(0));  
    vector<Point> polygon_shift;
    matrix_to_points(polys_close_shift,polygon_shift);
    fillConvexPoly(raw_img_cp8,&polygon_shift[0], new_polys_close.rows(), Scalar(255,0,0), 0,0);    //faster than point in polygon test.
    vector<Point2i> locations_shift;   // output, locations of non-zero pixels 
    findNonZero(raw_img_cp8, locations_shift);  
    
    inside_pts_homo_out.resize(3,locations_shift.size());
    part_polygon_bgr.resize(3,locations_shift.size());
    cv::Vec3b bgr_value;
    for (int ind=0;ind<locations_shift.size();ind++)
    {
	if (downsample_poly){
	    inside_pts_homo_out(0,ind)=(locations_shift[ind].x+bounding_box.x)*2;
	    inside_pts_homo_out(1,ind)=(locations_shift[ind].y+bounding_box.y)*2;
	    inside_pts_homo_out(2,ind)=1;
	}
	else{
	    inside_pts_homo_out(0,ind)=locations_shift[ind].x+bounding_box.x;
	    inside_pts_homo_out(1,ind)=locations_shift[ind].y+bounding_box.y;
	    inside_pts_homo_out(2,ind)=1;  
	}
	bgr_value=bgr_img.at<cv::Vec3b>(inside_pts_homo_out(1,ind),inside_pts_homo_out(0,ind));  //at row (y) and col(x)
	part_polygon_bgr(0,ind)=bgr_value[2];   //r
	part_polygon_bgr(1,ind)=bgr_value[1];   //g
	part_polygon_bgr(2,ind)=bgr_value[0];   //b
    }
  //   inside_pts_homo_out.colwise() += Vector3f(bounding_box.x,bounding_box.y,0); 
}


void popup_plane::find_2d_3d_closed_polygon(const Matrix4f& transToWolrd)
{
   Matrix4f invT=transToWolrd.inverse();
  //   Vector<Vector4f> direc(ground_seg2d_lines_connect.rows());    //draw a vector in world, and project to camera
    MatrixXf up_direc(ground_seg2d_lines_connect.rows(),4);
    MatrixXf down_direc(ground_seg2d_lines_connect.rows(),4);
    for (int line_ind=0;line_ind<ground_seg2d_lines_connect.rows();line_ind++)  
    {
	for (int pt_ind=0;pt_ind<2;pt_ind++)
	{
	    MatrixXf vertical_3d_world(3,2); // each column is point
	    Vector3f temp_vertex=ground_seg3d_lines_world.row(line_ind).segment<3>(pt_ind*3);
	    vertical_3d_world.col(0)=temp_vertex;     //.row(line_ind).segment<3>(pt_ind*3);
	    vertical_3d_world.col(1)=temp_vertex+Vector3f(0,0,2);
	    MatrixXf vertical_3d_sensor=homo_to_real_coord(invT*real_to_homo_coord(vertical_3d_world));
	    MatrixXf vertical_2d_image=homo_to_real_coord(Kalib*vertical_3d_sensor);
	    Vector2f temp_dir=vertical_2d_image.col(1)-vertical_2d_image.col(0);
	    if (temp_dir(1)>0)
	      temp_dir=-temp_dir;
	    up_direc.row(line_ind).segment<2>(pt_ind*2)=temp_dir;
	    down_direc.row(line_ind).segment<2>(pt_ind*2)=-temp_dir;
	}
    }
    
    MatrixXf boudaryies_hits=MatrixXf::Zero(ground_seg2d_lines_connect.rows(),ground_seg2d_lines_connect.cols());
    MatrixXf boudaryies_hits_down=MatrixXf::Zero(ground_seg2d_lines_connect.rows(),ground_seg2d_lines_connect.cols());
    for (int line_ind=0;line_ind<ground_seg2d_lines_connect.rows();line_ind++)    
	for (int pt_ind=0;pt_ind<2;pt_ind++)      
	{
	    boudaryies_hits.row(line_ind).segment<2>(pt_ind*2)=direction_hit_boundary(ground_seg2d_lines_connect.row(line_ind).segment<2>(pt_ind*2),up_direc.row(line_ind).segment<2>(pt_ind*2));
	    boudaryies_hits_down.row(line_ind).segment<2>(pt_ind*2)=direction_hit_boundary(ground_seg2d_lines_connect.row(line_ind).segment<2>(pt_ind*2),down_direc.row(line_ind).segment<2>(pt_ind*2));
	}

  // find ground closed polygons
    int num_seg=ground_seg2d_lines_connect.rows();
    vector<MatrixXf> all_closed_polygons_out(num_seg+1);  // each element is a polygon n*2. each row of matrixxd is a vertex
    MatrixXf ground_seg2d_closed=ground_seg2d_lines_connect.leftCols(2);
    vert_stack_v_self(ground_seg2d_closed,ground_seg2d_lines_connect.row(num_seg-1).tail(2));  
    
    int seg=num_seg-1;  //considering the last segment
    Vector2f begin_hit_bound=boudaryies_hits_down.row(seg).head(2);
    Vector2f end_hit_bound=boudaryies_hits_down.row(seg).tail(2);
    if (ground_seg2d_lines_connect(seg,3)==height-1 && ground_seg2d_lines_connect(seg,2)<width-1)  {}  
    else if (ground_seg2d_lines_connect(seg,2)==width-1 && ground_seg2d_lines_connect(seg,3)<=height-1)  
	  vert_stack_v_self(ground_seg2d_closed,Vector2f(width-1,height-1));  
    else{
	  if (end_hit_bound(0)==width-1){
		vert_stack_v_self(ground_seg2d_closed,end_hit_bound);
		vert_stack_v_self(ground_seg2d_closed,Vector2f(width-1,height-1));
	  }
	  if (0<=end_hit_bound(0) & end_hit_bound(0)<width-1)
	      vert_stack_v_self(ground_seg2d_closed,end_hit_bound);
    }
      
    seg=0;  //considering the last segment
    begin_hit_bound=boudaryies_hits_down.row(seg).head(2);
    end_hit_bound=boudaryies_hits_down.row(seg).tail(2);
    if (ground_seg2d_lines_connect(seg,1)==height-1 && ground_seg2d_lines_connect(seg,0)>0)  {}  
    else if (ground_seg2d_lines_connect(seg,0)==0 && ground_seg2d_lines_connect(seg,1)<=height-1)  
	  vert_stack_v_self(ground_seg2d_closed,Vector2f(0,height-1));  
    else{
	if (begin_hit_bound(0)==width-1){
	    vert_stack_v_self(ground_seg2d_closed,Vector2f(0,height-1));  
	    vert_stack_v_self(ground_seg2d_closed,end_hit_bound);	 
	}
	if (0<=begin_hit_bound(0) & begin_hit_bound(0)<width-1)
	    vert_stack_v_self(ground_seg2d_closed,end_hit_bound);
    }

    vert_stack_v_self(ground_seg2d_closed,ground_seg2d_closed.row(0));
    all_closed_polygons_out[0]=ground_seg2d_closed;  
    
  // find wall closed polygons
    for(seg=0;seg<num_seg;seg++)
    {
      MatrixXf_row line_vertexs= ground_seg2d_lines_connect.row(seg);
      line_vertexs.resize(2,2);
      MatrixXf part_wall_close_polys=line_vertexs;
      
      begin_hit_bound=boudaryies_hits.row(seg).head(2);
      end_hit_bound=boudaryies_hits.row(seg).tail(2);
      if (seg==num_seg-1)   // last segment  rightmost wall
      {
	  if (ground_seg2d_lines_connect(seg,3)==height-1 && ground_seg2d_lines_connect(seg,2)<width-1){
	      vert_stack_v_self(part_wall_close_polys,Vector2f(width-1, height-1)); 
	      
	      if (begin_hit_bound(0)==width-1)
		  vert_stack_v_self(part_wall_close_polys,begin_hit_bound);
	      if (0<begin_hit_bound(0) && begin_hit_bound(0)<width-1) {
		  vert_stack_v_self(part_wall_close_polys,Vector2f(width-1, 0));
		  vert_stack_v_self(part_wall_close_polys,begin_hit_bound);
	      }
	      if (begin_hit_bound(0)==0) {
		  vert_stack_v_self(part_wall_close_polys,Vector2f(width-1, 0));
		  vert_stack_v_self(part_wall_close_polys,Vector2f(0, 0));
		  vert_stack_v_self(part_wall_close_polys,begin_hit_bound);
	      }
	  }
	  else if (ground_seg2d_lines_connect(seg,2)==width-1 && ground_seg2d_lines_connect(seg,3)<=height-1){            
	      if (begin_hit_bound(0)==width-1)
		  vert_stack_v_self(part_wall_close_polys,begin_hit_bound);
	      if (0<begin_hit_bound(0) && begin_hit_bound(0)<width-1) {
		  vert_stack_v_self(part_wall_close_polys,Vector2f(width-1, 0));
		  vert_stack_v_self(part_wall_close_polys,begin_hit_bound);
	      }
	      if (begin_hit_bound(0)==0) {
		  vert_stack_v_self(part_wall_close_polys,Vector2f(width-1, 0));
		  vert_stack_v_self(part_wall_close_polys,Vector2f(0, 0));
		  vert_stack_v_self(part_wall_close_polys,begin_hit_bound);
	      }
	  }
	  else{
	      if (begin_hit_bound(0)==width-1 && end_hit_bound(0)==width-1) {
		  vert_stack_v_self(part_wall_close_polys,end_hit_bound);
		  vert_stack_v_self(part_wall_close_polys,begin_hit_bound);
	      }
	      if (0<begin_hit_bound(0) && begin_hit_bound(0)<width-1 && end_hit_bound[0]==width-1){
		  vert_stack_v_self(part_wall_close_polys,end_hit_bound);
		  vert_stack_v_self(part_wall_close_polys,Vector2f(width-1, 0));
		  vert_stack_v_self(part_wall_close_polys,begin_hit_bound);
	      }
	      if (0<begin_hit_bound(0) && begin_hit_bound(0)<width-1 && 0<end_hit_bound(0) && end_hit_bound(0)<width-1) {
		  vert_stack_v_self(part_wall_close_polys,end_hit_bound);
		  vert_stack_v_self(part_wall_close_polys,begin_hit_bound);
	      }
	      if (begin_hit_bound(0)==0 && 0<end_hit_bound(0) && end_hit_bound(0)<width-1) {
		  vert_stack_v_self(part_wall_close_polys,end_hit_bound);
		  vert_stack_v_self(part_wall_close_polys,Vector2f(0,0));
		  vert_stack_v_self(part_wall_close_polys,begin_hit_bound);
	      }
	      if (begin_hit_bound(0)==0  && end_hit_bound(0)==0) {
		  vert_stack_v_self(part_wall_close_polys,end_hit_bound);
		  vert_stack_v_self(part_wall_close_polys,begin_hit_bound);
	      }
	  }
      }
      else if (0<seg && seg<num_seg-1)  // middle segments
      {	    
	      if (begin_hit_bound(0)==width-1 && end_hit_bound(0)==width-1) {
		  vert_stack_v_self(part_wall_close_polys,end_hit_bound);
		  vert_stack_v_self(part_wall_close_polys,begin_hit_bound);
	      }
	      if (0<begin_hit_bound(0) && begin_hit_bound(0)<width-1 && end_hit_bound[0]==width-1){
		  vert_stack_v_self(part_wall_close_polys,end_hit_bound);
		  vert_stack_v_self(part_wall_close_polys,Vector2f(width-1, 0));
		  vert_stack_v_self(part_wall_close_polys,begin_hit_bound);
	      }
	      if (0<begin_hit_bound(0) && begin_hit_bound(0)<width-1 && 0<end_hit_bound(0) && end_hit_bound(0)<width-1) {
		  vert_stack_v_self(part_wall_close_polys,end_hit_bound);
		  vert_stack_v_self(part_wall_close_polys,begin_hit_bound);
	      }
	      if (begin_hit_bound(0)==0 && 0<end_hit_bound(0) && end_hit_bound(0)<width-1) {
		  vert_stack_v_self(part_wall_close_polys,end_hit_bound);
		  vert_stack_v_self(part_wall_close_polys,Vector2f(0,0));
		  vert_stack_v_self(part_wall_close_polys,begin_hit_bound);
	      }
	      if (begin_hit_bound(0)==0  && end_hit_bound(0)==0) {
		  vert_stack_v_self(part_wall_close_polys,end_hit_bound);
		  vert_stack_v_self(part_wall_close_polys,begin_hit_bound);
	      }
      }
      else  // leftmost wall
      {        
	  if (ground_seg2d_lines_connect(seg,1)==height-1 && ground_seg2d_lines_connect(seg,0)>0){
	      if (end_hit_bound(0)==width-1) {
		  vert_stack_v_self(part_wall_close_polys,end_hit_bound);
		  vert_stack_v_self(part_wall_close_polys,Vector2f(width-1, 0));
		  vert_stack_v_self(part_wall_close_polys,Vector2f(0, 0));	    
	      }
	      if (0<end_hit_bound(0) && end_hit_bound(0)<width-1) {
		  vert_stack_v_self(part_wall_close_polys,end_hit_bound);
		  vert_stack_v_self(part_wall_close_polys,Vector2f(0, 0));
	      }
	      if (end_hit_bound(0)==0)
		  vert_stack_v_self(part_wall_close_polys,end_hit_bound);
	      
	      vert_stack_v_self(part_wall_close_polys,Vector2f(0, height-1));
	  }
	  else if (ground_seg2d_lines_connect(seg,0)==0 && ground_seg2d_lines_connect(seg,1)<=height-1){
	      if (end_hit_bound(0)==width-1) {
		  vert_stack_v_self(part_wall_close_polys,end_hit_bound);
		  vert_stack_v_self(part_wall_close_polys,Vector2f(width-1, 0));
		  vert_stack_v_self(part_wall_close_polys,Vector2f(0, 0));	    
	      }
	      if (0<end_hit_bound(0) && end_hit_bound(0)<width-1) {
		  vert_stack_v_self(part_wall_close_polys,end_hit_bound);
		  vert_stack_v_self(part_wall_close_polys,Vector2f(0, 0));
	      }
	      if (end_hit_bound(0)==0)
		  vert_stack_v_self(part_wall_close_polys,end_hit_bound);
	  }
	  else{
	      if (begin_hit_bound(0)==width-1 && end_hit_bound(0)==width-1) {
		  vert_stack_v_self(part_wall_close_polys,end_hit_bound);
		  vert_stack_v_self(part_wall_close_polys,begin_hit_bound);
	      }
	      if (0<begin_hit_bound(0) && begin_hit_bound(0)<width-1 && end_hit_bound[0]==width-1){
		  vert_stack_v_self(part_wall_close_polys,end_hit_bound);
		  vert_stack_v_self(part_wall_close_polys,Vector2f(width-1, 0));
		  vert_stack_v_self(part_wall_close_polys,begin_hit_bound);
	      }
	      if (0<begin_hit_bound(0) && begin_hit_bound(0)<width-1 && 0<end_hit_bound(0) && end_hit_bound(0)<width-1) {
		  vert_stack_v_self(part_wall_close_polys,end_hit_bound);
		  vert_stack_v_self(part_wall_close_polys,begin_hit_bound);
	      }
	      if (begin_hit_bound(0)==0 && 0<end_hit_bound(0) && end_hit_bound(0)<width-1) {
		  vert_stack_v_self(part_wall_close_polys,end_hit_bound);
		  vert_stack_v_self(part_wall_close_polys,Vector2f(0,0));
		  vert_stack_v_self(part_wall_close_polys,begin_hit_bound);
	      }
	      if (begin_hit_bound(0)==0  && end_hit_bound(0)==0) {
		  vert_stack_v_self(part_wall_close_polys,end_hit_bound);
		  vert_stack_v_self(part_wall_close_polys,begin_hit_bound);
	      }
	  }
      }
      vert_stack_v_self(part_wall_close_polys,ground_seg2d_lines_connect.row(seg).head(2).transpose());
      all_closed_polygons_out[seg+1]=part_wall_close_polys;    
    }

    // compute the bounded polygons (considering the ceiling.)
    all_closed_2d_bound_polygons.resize(all_closed_polygons_out.size());
    all_3d_bound_polygons_world.resize(all_closed_polygons_out.size());  
    for (int ii=0;ii<all_closed_polygons_out.size();ii++)  
    {      
  //       cout<<"polygons "<<all_closed_polygons_out[ii]<<endl;
	all_closed_2d_bound_polygons[ii]=all_closed_polygons_out[ii].transpose();  
	MatrixXf plane_seg_ray=invK*real_to_homo_coord(all_closed_2d_bound_polygons[ii]);
	MatrixXf plane_seg_3d_sensor;
	ray_plane_interact(plane_seg_ray, all_planes_sensor.row(ii),plane_seg_3d_sensor);
	
	//TODO restrict a plane not to reach very far in camera z direction, at most 10 meters;  this doesn't change the point cloud visualization
	if (ii>0){  // if not ground
// 	    Vector3f bgn_sensor_pt=plane_seg_3d_sensor.col(0);
// 	    Vector3f end_sensor_pt=plane_seg_3d_sensor.col(1);
// 	    if (bgn_sensor_pt(2)<10 && end_sensor_pt(2)>10)
	  
	    MatrixXf plane_seg_3d_world = homo_to_real_coord(transToWolrd*real_to_homo_coord(plane_seg_3d_sensor));  //3*n  
	    //TODO cut a fixed height // not that straight forward to cut a suitable polygon. later, if I can segment ceiling. it works
	    MatrixXf cut_plane_seg_3d_world(3,2); 
	    cut_plane_seg_3d_world.col(0)=plane_seg_3d_world.col(0); //first segment is on the ground
	    cut_plane_seg_3d_world.col(1)=plane_seg_3d_world.col(1);
	    for (int pt_ind=0;pt_ind<plane_seg_3d_world.cols();pt_ind++)
	    {
		if (plane_seg_3d_world(2,pt_ind)<=ceiling_threshold)
		      hori_stack_v_self(cut_plane_seg_3d_world,plane_seg_3d_world.col(pt_ind));
		else{  // need to find intersection of a line segments with threshold height
		      if (plane_seg_3d_world(2,pt_ind-1)<ceiling_threshold) {
			  Vector3f intersec;
			  if (points_intersect_plane(plane_seg_3d_world.col(pt_ind-1),plane_seg_3d_world.col(pt_ind),ceiling_threshold,intersec))
			    hori_stack_v_self(cut_plane_seg_3d_world,intersec);
		      }
		      if (plane_seg_3d_world(2,pt_ind+1)<ceiling_threshold) {
			  Vector3f intersec;
			  if (points_intersect_plane(plane_seg_3d_world.col(pt_ind),plane_seg_3d_world.col(pt_ind+1),ceiling_threshold,intersec))
			    hori_stack_v_self(cut_plane_seg_3d_world,intersec);
		      }
		}
	    }
	    all_3d_bound_polygons_world[ii]=cut_plane_seg_3d_world; // if don't need cutting ceiling, just use  plane_seg_3d_world
       }
       
       if (ii==0){  // if ground plane
	    //TODO could use good plane indice, to select edges otherwise might contain some bad points, front or back
       }
      
    }
}

void popup_plane::get_plane_equation(const Matrix4f& transToWolrd, const MatrixXf& ground_seg2d_lines_connect_cp)
{
    total_plane_number=ground_seg2d_lines_connect_cp.rows()+1;
    ground_seg2d_lines_connect=ground_seg2d_lines_connect_cp;
    if (total_plane_number>1)  //
    {         
        Vector4f ground_plane_world(0,0,-1,0);  //plane parameters for ground in world(footprint), treated as column vector. normal is away from camera
	Vector4f ground_plane_sensor=transToWolrd.transpose()*ground_plane_world;

	MatrixXf ground_seg2d_homo(ground_seg2d_lines_connect.rows()*2,3); // n*4
	MatrixXf_row temp=ground_seg2d_lines_connect;  // row order for resize
	temp.resize(ground_seg2d_lines_connect.rows()*2,2);  //2n*2
	ground_seg2d_homo<<temp.array(),VectorXf::Ones(ground_seg2d_lines_connect.rows()*2); // 2n*3	
	MatrixXf ground_seg_ray=invK*ground_seg2d_homo.transpose();    //each column is a 3D world coordinate  3*n    	
	MatrixXf ground_seg3d_sensor;
	ray_plane_interact(ground_seg_ray,ground_plane_sensor,ground_seg3d_sensor);
	MatrixXf ground_seg3d_homo_sensor;
	real_to_homo_coord(ground_seg3d_sensor,ground_seg3d_homo_sensor);  //3*n
	
	MatrixXf ground_seg3d_homo_world=transToWolrd*ground_seg3d_homo_sensor;//   # compute world ground polygons  
	homo_to_real_coord(ground_seg3d_homo_world,ground_seg3d_lines_world); // #3*n 
	
	temp=ground_seg3d_sensor.transpose();
	temp.resize(ground_seg3d_sensor.cols()/2,6); // n/2 * 6
	ground_seg3d_lines_sensor=temp;

	temp=ground_seg3d_lines_world.transpose();
	temp.resize(ground_seg3d_lines_world.cols()/2,6); // n/2 * 6
	ground_seg3d_lines_world=temp;

// 	cout<<"ground_seg3d_lines_sensor "<<ground_seg3d_lines_sensor<<endl;
	// compute bouned plane distance to camera. and find close planes as good plane
	all_plane_dist_to_cam.resize(ground_seg3d_lines_world.rows()+1);
	all_plane_dist_to_cam(0)=transToWolrd(2,3);  // height
	Vector2f camera_pose=transToWolrd.col(3).head(2);
	vector<int> close_front_plane_inds(1);
	close_front_plane_inds[0]=0;
	for (int seg=0;seg<ground_seg3d_lines_world.rows();seg++)
	{	    
	    Vector2f begin_pt=ground_seg3d_lines_world.row(seg).segment<2>(0);
	    Vector2f end_pt=ground_seg3d_lines_world.row(seg).segment<2>(3);
	    all_plane_dist_to_cam(seg+1)=point_dist_lineseg(begin_pt,end_pt,camera_pose);
	    if ( (ground_seg3d_lines_sensor(seg,2)>0) && (ground_seg3d_lines_sensor(seg,5)>0))  // in front of the camera
// 	      if ( (ground_seg3d_lines_sensor(seg,2)<20) && (ground_seg3d_lines_sensor(seg,5)<20))  // not to far in z
// 		if  ((Vector2f(ground_seg3d_lines_sensor(seg,3),ground_seg3d_lines_sensor(seg,5))).norm()<25)
		if (all_plane_dist_to_cam(seg+1)<plane_cam_dist_thre)  // distance threshold
		  if (check_element_in_vector(float(seg+1),good_plane_indices))  // if it is not manually connected edges
		      close_front_plane_inds.push_back(seg+1);
	}
	good_plane_number=close_front_plane_inds.size();
	good_plane_indices.resize(good_plane_number);
	for (int i=0;i<good_plane_number;i++)
	    good_plane_indices(i)=close_front_plane_inds[i];

	//compute wall plane parameters in world frame  
	int num_seg=ground_seg2d_lines_connect.rows();
	
	all_planes_world.resize(num_seg+1,4);   // store all the plane parameters in world frame
	all_planes_sensor.resize(num_seg+1,4);  // store all the plane parameters in sensor frame
	all_planes_world.row(0)=ground_plane_world;
	all_planes_sensor.row(0)=ground_plane_sensor;
	
	for (int seg=0;seg<num_seg;seg++)
	{
	    Vector3f partwall_line3d_world_bg=ground_seg3d_lines_world.row(seg).head(3);
	    Vector3f partwall_line3d_world_en=ground_seg3d_lines_world.row(seg).tail(3);
	    Vector3f temp1=partwall_line3d_world_en-partwall_line3d_world_bg;
	    Vector3f temp2=ground_plane_world.head(3);
	    Vector3f partwall_normal_world=temp1.cross(temp2);
	    float dist=-partwall_normal_world.transpose()*partwall_line3d_world_bg;
	    Vector4f partwall_plane_world;   partwall_plane_world<<partwall_normal_world,dist;
	    Vector4f partwall_plane_sensor=transToWolrd.transpose()*partwall_plane_world;// wall plane in sensor frame    
	    
	    all_planes_world.row(seg+1)=partwall_plane_world;
	    all_planes_sensor.row(seg+1)=partwall_plane_sensor;
	}
	ceiling_plane_world<<0,0,-1,ceiling_threshold;
	ceiling_plane_sensor=transToWolrd.transpose()*ceiling_plane_world;
	find_2d_3d_closed_polygon(transToWolrd);  //also find polygon
    }
    else    
    {
      total_plane_number=0;
      good_plane_number=0;
      cout<<"pop_up_wall: cannot compute plane 3D polygons!!!"<<endl;    
    }
}


void popup_plane::get_ground_edges(const cv::Mat &raw_img, const cv::Mat &label_map)
{
    width=label_map.cols;
    height=label_map.rows;    

    cv::Mat gray_img;
    if ( raw_img.channels()==3 )
	cv::cvtColor(raw_img, gray_img, CV_BGR2GRAY);
    else
	gray_img=raw_img;        
    
    cv::Mat contrast_img=gray_img;
    cv::Mat lsd_edge_mat;
    cv::Scalar mean_intensity=cv::mean(gray_img);
    if (mean_intensity[0]<90)  // if very dark
	cv::equalizeHist(gray_img, contrast_img);

    std::clock_t begin2 = clock();    
//     void line_lbd_detect::detect_lines(Mat& gray_img_mat, cv::Mat& lines_mat)
    line_lbd_ptr->use_LSD=true;
    if (use_fast_edges_detection)
	line_lbd_ptr->detect_raw_lines(contrast_img,lsd_edge_mat,true);
    else
	line_lbd_ptr->detect_raw_lines(contrast_img,lsd_edge_mat,false);  // 3ms  using lbd algorithms
//     std::cout<<"detect edges time  "<< double(clock() - begin2) / CLOCKS_PER_SEC<<std::endl;
    
    std::clock_t begin3 = clock();
    VectorXf open_walls_in_close_ind;  // open polygons id in close polygons  matrix to vector
    if (inited_python)
	std::tie(ground_seg2d_lines_open, ground_seg2d_lines_connect, open_walls_in_close_ind)\
	     =edge_get_polygons(lsd_edge_mat,label_map);  // using python function to select the 2d segments

//     cout<<"python return edges "<<ground_seg2d_lines_open<<endl;
//     std::cout<<"select edges time  "<< double(clock() - begin3) / CLOCKS_PER_SEC<<std::endl;
    if (ground_seg2d_lines_connect.size()>0)   // if there is any gound edges
    {
	VectorXf good_plane_indices_temp(open_walls_in_close_ind.rows()+1);
	good_plane_indices_temp<<0, open_walls_in_close_ind.array()+1;
	good_plane_indices=good_plane_indices_temp;
	total_plane_number=ground_seg2d_lines_connect.rows()+1;
	good_plane_number=good_plane_indices.rows();
    }
    else   // if there is no ground edges, then ground plane has no meaning as there is no boundary for the ground plane
    {
	good_plane_indices.resize(0);
	total_plane_number=0;
	good_plane_number=0;
	cout<<"cannot find ground edges!!!"<<endl;
    }
}


void popup_plane::generate_cloud(const cv::Mat &bgr_img, const Matrix4f& transToWolrd, bool downsample_poly, bool separate_cloud,
				 bool mixed_cloud,bool get_label_img,float depth_thre)
{
   if (total_plane_number>0)
   {      
      vector<MatrixX8uu> partplane_rgb_all(good_plane_number);
      vector<MatrixXf> partplane_pts_sensor_all(good_plane_number);
      vector<MatrixXf> partplane_pts_world_all(good_plane_number);
      vector<MatrixXf> partplane_pts_2d_homo(good_plane_number); // 3*m homo point      
      if (get_label_img)
	  edge_label_img=cv::Mat(Size(width,height),CV_8UC1,Scalar(127)); // initialized as void
      for (int i=0;i<good_plane_indices.rows();i++)  //0 is ground
      {
	  int total_plane_id=good_plane_indices(i);
	  closed_polygons_homo_pts(all_closed_2d_bound_polygons[total_plane_id].transpose(),partplane_pts_2d_homo[i],bgr_img,partplane_rgb_all[i],downsample_poly);   //3*m  ~3ms
	  MatrixXf partwall_pts_ray=invK*partplane_pts_2d_homo[i];
	  Vector4f partwall_plane_sensor=all_planes_sensor.row(total_plane_id);  // plane parameters in sensor frame
	  ray_plane_interact(partwall_pts_ray,partwall_plane_sensor,partplane_pts_sensor_all[i]);  //3*m  3-4ms
	  MatrixXf partplane_3d_homo_sensor;
	  real_to_homo_coord(partplane_pts_sensor_all[i],partplane_3d_homo_sensor);
	  homo_to_real_coord(transToWolrd*partplane_3d_homo_sensor,partplane_pts_world_all[i]);  //3*n
	  
	  if (get_label_img){   // generate ground label image
	    for (int pt=0;pt<partplane_pts_2d_homo[i].cols();pt++){
	      if (total_plane_id==0)  // ground
		edge_label_img.at<uchar>(partplane_pts_2d_homo[i](1,pt),partplane_pts_2d_homo[i](0,pt))=255; 
	      else    // wall
		edge_label_img.at<uchar>(partplane_pts_2d_homo[i](1,pt),partplane_pts_2d_homo[i](0,pt))=0; 
	    }
	  }
      }
      if (separate_cloud | mixed_cloud)
	  matrixToCloud(partplane_pts_sensor_all,partplane_pts_world_all, partplane_rgb_all,separate_cloud,mixed_cloud,depth_thre);
      
      if (get_label_img & downsample_poly){ // if downsample polygon, there mayby many void pixels
	  cv::resize(edge_label_img,edge_label_img,cv::Size(),0.5,0.5,INTER_NEAREST);  //downsample raw image	    
	  cv::resize(edge_label_img,edge_label_img,cv::Size(),2,2,INTER_NEAREST);      //upsample raw image	
      }
   }
   else
   {
     pcl_cloud_world.reset(new pclRGB);
     partplane_clouds_open_all.resize(0);
     cout<<"cannot generate cloud!!!"<<endl;
   }
}


// // xyz rgb are all 3*m, will remove receiling point
void popup_plane::matrixToCloud(const vector<MatrixXf> &local_pts_parts, const vector<MatrixXf> &global_pts_parts, 
				const vector<MatrixX8uu> &rgb_parts,bool separate_cloud,bool mixed_cloud,float depth_thre)
{
    if(separate_cloud)
        partplane_clouds_open_all.resize(good_plane_number);  // note that vector resize will copy previous elements, not clear them.  eigen resize will clear all      
    if(mixed_cloud)
    {
	pcl_cloud_world.reset(new pclRGB);    
	int total_pt_num=0;
	for (int plane_id=0;plane_id<good_plane_number;plane_id++)
	    total_pt_num=total_pt_num+global_pts_parts[plane_id].cols();
	pcl_cloud_world->resize(total_pt_num);
    }
    
    pcl::PointXYZRGB pt;
    int total_pt_counter=-1;    
    for (int plane_id=0;plane_id<good_plane_number;plane_id++)
    {
	if (separate_cloud){
	    partplane_clouds_open_all[plane_id].reset(new pclRGB);
	    partplane_clouds_open_all[plane_id]->clear();
	    partplane_clouds_open_all[plane_id]->resize(global_pts_parts[plane_id].cols());
	}
	for (int ind=0;ind<global_pts_parts[plane_id].cols();ind++)
	{
	    total_pt_counter++;
	    if (local_pts_parts[plane_id](2,ind)<0)
	       continue;
	    if (local_pts_parts[plane_id](2,ind)>depth_thre)
	       continue;
// 	    if (Vector2f(local_pts_parts[plane_id](0,ind),local_pts_parts[plane_id](2,ind)).norm()>5)  # compute distance to camera, instead of z
// 	       continue;
	    if (global_pts_parts[plane_id](2,ind)<-0.2) // don't show underground pts
	      continue;
	    pt.x=global_pts_parts[plane_id](0,ind);
	    pt.y=global_pts_parts[plane_id](1,ind);
	    pt.z=global_pts_parts[plane_id](2,ind)<ceiling_threshold ? global_pts_parts[plane_id](2,ind):ceiling_threshold;
	    pt.r=rgb_parts[plane_id](0,ind);
	    pt.g=rgb_parts[plane_id](1,ind);
	    pt.b=rgb_parts[plane_id](2,ind);
	    if (separate_cloud)
	       partplane_clouds_open_all[plane_id]->points[ind]=pt;
	    if (mixed_cloud)
	       pcl_cloud_world->points[total_pt_counter]=pt;
	}
    }
}