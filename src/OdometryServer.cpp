// This file was heavily inspired by KISS-ICP, for this reason we report here
// the original LICENSE
// MIT License
//
// Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
// Stachniss.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "OdometryServer.hpp"
#include "rotation.hpp"

#include <Eigen/Core>
#include <condition_variable>
#include <csignal>
#include <filesystem>
#include <sys/stat.h>
#include <vector>

#include "Utils.hpp"
#include "kiss_icp/pipeline/KissICP.hpp"

// ROS
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>

#include "data_recorder.h"
#include <std_srvs/Trigger.h>

std::shared_ptr<DataRecorder<RecordPointType>> recorder_ptr_;
ros::ServiceServer recorder_server_;
std::string result_dir, dataset, data_id, test_topic, algorithm, param_set_name;
std::vector<std::string> lidar_names;
std::vector<int> lidar_indices;
int save_frame_mode = 0; 
std::string save_dir;

std::condition_variable sig_buffer_;

void SigHandle(int sig) {
  ROS_WARN("catch sig %d", sig);
  sig_buffer_.notify_all();
}

void recordRamUsage(double stamp)
{
    pid_t pid = getpid();
    std::string path = "/proc/" + std::to_string(pid) + "/status";
    std::ifstream file(path);
    std::string line;
    double mem_usage = 0.0;
    while (std::getline(file, line))
    {
        if (line.find("VmRSS:") == 0)
        {
            mem_usage = std::stod(line.substr(6)) / 1024.0; // Convert to MB
            break;
        }
    }

    recorder_ptr_->recordValue("RAM_usage", stamp, mem_usage);
}

// void recordCloud() {

//     if (!recorder_ptr_->isCloudRecordEnabled())
//         return;

//     if (!recorder_ptr_->isInit()) {
//         std::cout << "Recorder is not initialized!" << std::endl;
//         return;
//     }

//     pcl::PointCloud<RecordPointType>::Ptr record_cloud(new pcl::PointCloud<RecordPointType>);
//     for (size_t i = 0; i < feats_down_body->points.size(); i++) {
//         RecordPointType point;
//         point.x = feats_down_body->points[i].x;
//         point.y = feats_down_body->points[i].y;
//         point.z = feats_down_body->points[i].z;
//         point.intensity = feats_down_body->points[i].intensity;

//         record_cloud->push_back(point);
//     }

//     recorder_ptr_->recordCloud(record_cloud, lidar_end_time);
//     recorder_ptr_->saveCloud();
// }

bool data_recorder_callback(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res)
{
    if (recorder_ptr_ != nullptr)
    {

        recorder_ptr_->savePose();
        recorder_ptr_->saveTime();
        recorder_ptr_->saveValue();
        recorder_ptr_->saveStatus("Finished");

        res.success = true;
        res.message = "Data saved successfully.";
    }
    else
    {
        res.success = false;
        res.message = "Recorder pointer is null, cannot save data.";
    }
    return true;
}

namespace lio_ekf {

OdometryServer::OdometryServer(const ros::NodeHandle &nh,
                               const ros::NodeHandle &pnh)
    : nh_(nh), pnh_(pnh) {

  nh_.param<bool>("lidar/deskew", lio_para_.deskew, false);
  nh_.param<bool>("lidar/preprocess", lio_para_.preprocess, true);
  nh_.param<float>("lidar/max_range", lio_para_.max_range, 100.0);
  nh_.param<float>("lidar/min_range", lio_para_.min_range, 5.0);
  nh_.param<int>("lidar/max_points_per_voxel", lio_para_.max_points_per_voxel,
                 20);

  nh_.param<float>("lidar/voxel_size", lio_para_.voxel_size, 1);

  nh_.param<int>("lidar/max_iteration", lio_para_.max_iteration, 1);
  // common
  nh_.param<std::string>("common/lidar_topic", lid_topic, "");
  nh_.param<std::string>("common/imu_topic", imu_topic, "");

  nh_.getParam("outputdir", outputdir);

  // imu
  std::vector<double> arw, vrw, gyrbias_std, accbias_std;

  nh_.param<std::vector<double>>("imu/arw", arw, std::vector<double>());
  nh_.param<std::vector<double>>("imu/vrw", vrw, std::vector<double>());
  nh_.param<std::vector<double>>("imu/gbstd", gyrbias_std,
                                 std::vector<double>());
  nh_.param<std::vector<double>>("imu/abstd", accbias_std,
                                 std::vector<double>());
  nh_.param<double>("imu/corrtime", lio_para_.imunoise.correlation_time, 1);

  // std::vector to Eigen::vector

  memcpy(lio_para_.imunoise.angle_randomwalk.data(), &arw[0],
         3 * sizeof(double));
  memcpy(lio_para_.imunoise.velocity_randomwalk.data(), &vrw[0],
         3 * sizeof(double));
  memcpy(lio_para_.imunoise.gyrbias_std.data(), &gyrbias_std[0],
         3 * sizeof(double));
  memcpy(lio_para_.imunoise.accbias_std.data(), &accbias_std[0],
         3 * sizeof(double));

  // lio

  std::vector<double> initposstd, initvelstd, initattstd;
  std::vector<double> extrinsic_T, extrinsic_R, imu_tran_R;
  Eigen::Matrix3d lidar_imu_extrin_R;
  Eigen::Vector3d lidar_imu_extrin_T;

  nh_.param<std::vector<double>>("lio/initposstd", initposstd,
                                 std::vector<double>());
  nh_.param<std::vector<double>>("lio/initvelstd", initvelstd,
                                 std::vector<double>());
  nh_.param<std::vector<double>>("lio/initattstd", initattstd,
                                 std::vector<double>());
  nh_.param<std::vector<double>>("lio/extrinsic_T", extrinsic_T,
                                 std::vector<double>());
  nh_.param<std::vector<double>>("lio/extrinsic_R", extrinsic_R,
                                 std::vector<double>());
  nh_.param<std::vector<double>>("lio/imu_tran_R", imu_tran_R,
                                 std::vector<double>());

  memcpy(lio_para_.initstate_std.pos.data(), &initposstd[0],
         3 * sizeof(double));
  memcpy(lio_para_.initstate_std.vel.data(), &initvelstd[0],
         3 * sizeof(double));
  memcpy(lio_para_.initstate_std.euler.data(), &initattstd[0],
         3 * sizeof(double));
  memcpy(lidar_imu_extrin_T.data(), &extrinsic_T[0], 3 * sizeof(double));
  memcpy(lidar_imu_extrin_R.data(), &extrinsic_R[0], 9 * sizeof(double));
  memcpy(lio_para_.imu_tran_R.data(), &imu_tran_R[0], 9 * sizeof(double));

  // the extrinsic lidar-inertial parameters before transforming the imu frame
  lio_para_.Trans_lidar_imu_origin.block<3, 3>(0, 0) = lidar_imu_extrin_R;
  lio_para_.Trans_lidar_imu_origin.block<3, 1>(0, 3) = lidar_imu_extrin_T;
  // imu frame has been changed
  lidar_imu_extrin_R = lio_para_.imu_tran_R * lidar_imu_extrin_R;
  lidar_imu_extrin_T = lio_para_.imu_tran_R * lidar_imu_extrin_T;
  lio_para_.Trans_lidar_imu.block<3, 3>(0, 0) = lidar_imu_extrin_R;
  lio_para_.Trans_lidar_imu.block<3, 1>(0, 3) = lidar_imu_extrin_T;

  lio_para_.initstate_std.euler *= D2R;

  lio_para_.imunoise.angle_randomwalk *= (D2R / 60.0);
  lio_para_.imunoise.velocity_randomwalk /= 60.0;
  lio_para_.imunoise.gyrbias_std *= (D2R / 3600.0);
  lio_para_.imunoise.accbias_std *= 1e-5;
  lio_para_.imunoise.correlation_time *= 3600;

  lio_para_.initstate_std.imuerror.gyrbias = lio_para_.imunoise.gyrbias_std;
  lio_para_.initstate_std.imuerror.accbias = lio_para_.imunoise.accbias_std;

  //----------------------------------------------------------------------------------------------------
  // Data Recorder Configurations
  nh.param<std::string>("data_recorder/result_dir", result_dir, "/");
  nh.param<std::string>("data_recorder/dataset", dataset, "dataset");
  nh.param<std::string>("data_recorder/data_id", data_id, "data_id");
  nh.param<std::string>("data_recorder/test_topic", test_topic, "test_topic");
  nh.param<std::string>("data_recorder/algorithm", algorithm, "fastlio");
  nh.param<std::string>("data_recorder/param_set_name", param_set_name, "default");
  nh.param<std::vector<std::string>>("data_recorder/lidar_names", lidar_names,
                                      std::vector<std::string>());
  nh.param<std::vector<int>>("data_recorder/lidar_indices", lidar_indices, std::vector<int>());
  nh.param<int>("data_recorder/save_frame_mode", save_frame_mode, 0);

  std::string lidars_combination = "";
  for (auto &index : lidar_indices)
  {
      lidars_combination += lidar_names[index] + "_";
  }
  lidars_combination = lidars_combination.substr(0, lidars_combination.size() - 1);

  save_dir = result_dir + "/" + dataset + "/" + data_id + "/" + test_topic + "/" + lidars_combination
              + "/" + algorithm  + "/" + param_set_name;

  // Check variables
  std::cout << "\033[32m" << "Data Recorder Configurations:" << std::endl;
  std::cout << "Result Directory: " << result_dir << std::endl;
  std::cout << "Data ID: " << data_id << std::endl;
  std::cout << "Test Topic: " << test_topic << std::endl;
  std::cout << "Parameter Set Name: " << param_set_name << std::endl;
  std::cout << "LiDAR Comb.: " << lidars_combination << std::endl;
  std::cout << "Save Directory: " << save_dir << std::endl;
  std::cout << "\033[0m" << std::endl;

  recorder_ptr_.reset(new DataRecorder<RecordPointType>());
  recorder_ptr_->init(save_dir, save_frame_mode, true);

  recorder_server_ = nh_.advertiseService("save_data", data_recorder_callback);
  //----------------------------------------------------------------------------------------------------


  lio_ekf_ = lio_ekf::LIOEKF(lio_para_);
  lio_ekf_.init();

  try {
    std::filesystem::create_directories(outputdir);
  } catch (std::exception &e) {
    ROS_ERROR_STREAM("Output folder open failed!");
  }

  std::string odomoutputpath = outputdir + "odo.txt";
  std::string odomoutputpath_tum = outputdir + "odo_tum.txt";

  odomRes_.open(odomoutputpath.c_str());
  odomRes_tum_.open(odomoutputpath_tum.c_str());

  if (odomRes_.is_open()) {
    ROS_WARN("Odometry output file opened!");
  } else {
    ROS_WARN("Cannot open odometry output file!");
  }

  odomRes_.setf(std::ios::fixed, std::ios::floatfield);
  odomRes_.precision(10);

  odomRes_tum_ << "# timestamp_s tx ty tz qx qy qz qw" << std::endl;
  odomRes_tum_.setf(std::ios::fixed, std::ios::floatfield);

  // Intializee subscribers
  pointcloud_sub_ = nh_.subscribe<sensor_msgs::PointCloud2>(
      lid_topic, 1000, &OdometryServer::lidar_cbk, this);
  imu_sub_ = nh_.subscribe<sensor_msgs::Imu>(imu_topic, 10000,
                                             &OdometryServer::imu_cbk, this);

  // Intialize publishers
  odom_publisher_ = nh_.advertise<nav_msgs::Odometry>("odometry", queue_size_);
  frame_publisher_ =
      nh_.advertise<sensor_msgs::PointCloud2>("frame", queue_size_);
  kpoints_publisher_ =
      nh_.advertise<sensor_msgs::PointCloud2>("keypoints", queue_size_);
  map_publisher_ =
      nh_.advertise<sensor_msgs::PointCloud2>("local_map", queue_size_);

  // Intialize trajectory publisher
  path_msg_.header.frame_id = odom_frame_;
  traj_publisher_ = nh_.advertise<nav_msgs::Path>("trajectory", queue_size_);

  // Make sure the order of the data in the buffer, pop out from the buffer once
  // added.

  signal(SIGINT, SigHandle);

  ros::Rate rate(1000);
  bool status = ros::ok();
  while (status) {

    ros::spinOnce();
    if (!data_synced_) {
      if (!imu_buffer_.empty() && !lidar_buffer_.empty()) {
        if (!imu_buffer_.empty()) {

          lio_ekf_.addImuData(imu_buffer_, false);
        }
        if (!lidar_buffer_.empty()) {

          if (lio_ekf_.getLiDARtimestamp() < lio_ekf_.getImutimestamp()) {

            lio_ekf_.addLidarData(lidar_buffer_, lidar_time_buffer_,
                                  lidar_header_buffer_,
                                  points_per_scan_time_buffer_);
          }
        }

        if (lio_ekf_.getLiDARtimestamp() >= lio_ekf_.getImutimestamp()) {
          data_synced_ = true;
        }
      }
    } else {

      if (lidar_buffer_.empty())
        if (imu_buffer_.empty()) {
          continue;
        }

      if (lio_ekf_.getLiDARtimestamp() < lio_ekf_.getImutimestamp() &&
          !lidar_buffer_.empty()) {

        lio_ekf_.addLidarData(lidar_buffer_, lidar_time_buffer_,
                              lidar_header_buffer_,
                              points_per_scan_time_buffer_);
      }

      if (!imu_buffer_.empty() &&
          !lidar_buffer_.empty()) // make sure lidar data is already there!
      {
        lio_ekf_.addImuData(imu_buffer_, false);
        lio_ekf_.newImuProcess();
        if (lio_ekf_.lidar_updated_) {
          writeResults(odomRes_);
          publishMsgs();
          lio_ekf_.lidar_updated_ = false;
        }
      }
    }

    status = ros::ok();
    rate.sleep();
  }
}

void OdometryServer::imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) {

  sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));
  double timestamp = msg->header.stamp.toSec();

  mtx_buffer_.lock();

  lio_ekf::IMU imu_meas;
  imu_meas.timestamp = timestamp;

  imu_meas.dt = timestamp - last_timestamp_imu_;
  imu_meas.angular_velocity << msg->angular_velocity.x, msg->angular_velocity.y,
      msg->angular_velocity.z;

  imu_meas.linear_acceleration << msg->linear_acceleration.x,
      msg->linear_acceleration.y, msg->linear_acceleration.z;

  imu_meas.angular_velocity = lio_para_.imu_tran_R * imu_meas.angular_velocity;
  imu_meas.linear_acceleration =
      lio_para_.imu_tran_R * imu_meas.linear_acceleration;

  imu_buffer_.push_back(imu_meas);

  last_timestamp_imu_ = timestamp;

  mtx_buffer_.unlock();
  sig_buffer_.notify_all();
}

void OdometryServer::lidar_cbk(const sensor_msgs::PointCloud2ConstPtr &msg) {

  for (auto &field : msg->fields) {
    if (field.name == "time" || field.name == "t") {
      break;
    }
  }

  if (msg->header.stamp.toSec() < last_timestamp_lidar_) {
    ROS_ERROR("lidar loop back, clear buffer");
    lidar_buffer_.clear();
    lidar_time_buffer_.clear();
    lidar_header_buffer_.clear();
  }

  mtx_buffer_.lock();

  // get timestamps for every points
  const auto &timestamps = [&]() -> std::vector<double> {
    if (!lio_para_.deskew)
      return {};
    return kiss_icp_ros::utils::GetTimestamps(msg);
  }();

  const auto &points = kiss_icp_ros::utils::PointCloud2ToEigen(msg);

  // deskew points

  lidar_buffer_.push_back(points);
  lidar_time_buffer_.push_back(msg->header.stamp.toSec());
  lidar_header_buffer_.push_back(msg->header);
  if (!timestamps.empty())
    points_per_scan_time_buffer_.push_back(timestamps);
  last_timestamp_lidar_ = msg->header.stamp.toSec();

  mtx_buffer_.unlock();
  sig_buffer_.notify_all();
}

void OdometryServer::writeResults(std::ofstream &odo) {
  lio_ekf::NavState navstate = lio_ekf_.getNavState();

  Eigen::MatrixXd curCov = lio_ekf_.getCovariance();

  const auto rotmat_imu = lio_para_.imu_tran_R.inverse();

  Eigen::Vector3d pos = rotmat_imu * navstate.pos;

  // transform pose from the used imu frame (front-right-down) to the original
  // imu frame
  Eigen::Matrix3d rotmat = lio_para_.imu_tran_R.inverse() *
                           Rotation::euler2matrix(navstate.euler) *
                           lio_para_.imu_tran_R;
  Eigen::Vector3d euler = Rotation::matrix2euler(rotmat);

  odo << lio_ekf_.getImutimestamp() << " " << pos.transpose() << " "
      << navstate.vel.transpose() << " " << euler.transpose() * R2D << " "
      << navstate.imuerror.gyrbias.transpose() << " "
      << navstate.imuerror.accbias.transpose() << std::endl;

  Eigen::Quaterniond quat = Rotation::euler2quaternion(euler);

  odomRes_tum_ << std::setprecision(18) << (lio_ekf_.getImutimestamp() / 1e9)
               << "e+09"
               << " " << std::setprecision(5) << pos[0] << " " << pos[1] << " "
               << pos[2] << " " << quat.x() << " " << quat.y() << " "
               << quat.z() << " " << quat.w() << std::endl;
}

void OdometryServer::publishMsgs() {
  // front-right-down to right-front-up for visualization
  Eigen::Matrix4d tmp = Eigen::Matrix4d::Identity();
  tmp.block<3, 3>(0, 0) = lio_para_.imu_tran_R.inverse();

  lio_ekf::NavState navstate = lio_ekf_.getNavState();

  Eigen::Matrix3d rotM = Rotation::euler2matrix(navstate.euler);
  Eigen::Matrix4d curpose = Eigen::Matrix4d::Identity();

  curpose.block<3, 3>(0, 0) = rotM;
  curpose.block<3, 1>(0, 3) = navstate.pos;

  Eigen::Matrix4d newpose = lio_ekf_.poseTran(tmp, curpose);

  newpose = lio_ekf_.poseTran(newpose, tmp);
  rotM = newpose.block<3, 3>(0, 0);
  Eigen::Quaterniond q_current = Rotation::matrix2quaternion(rotM);

  Eigen::Vector3d t_current = newpose.block<3, 1>(0, 3);

  Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
  pose.block<3, 3>(0, 0) = rotM;
  pose.block<3, 1>(0, 3) = t_current;
  Eigen::Matrix<double, 6, 6> pose_cov = lio_ekf_.getCovariance().block<6, 6>(0, 0);
  double imu_time  = lio_ekf_.getImutimestamp();
  recorder_ptr_->recordPose(imu_time, std::tie(pose, pose_cov));

  // Broadcast alias transformations to debug all datasets with the same
  // visualizer
  const auto original_pointcloud_frame = lio_ekf_.lidar_header_.frame_id;
  const auto stamp = lio_ekf_.lidar_header_.stamp;

  // Broadcast the tf
  geometry_msgs::TransformStamped transform_msg;
  transform_msg.header.stamp = stamp;
  transform_msg.header.frame_id = odom_frame_;
  transform_msg.child_frame_id = pointcloud_frame_;
  transform_msg.transform.rotation.x = q_current.x();
  transform_msg.transform.rotation.y = q_current.y();
  transform_msg.transform.rotation.z = q_current.z();
  transform_msg.transform.rotation.w = q_current.w();
  transform_msg.transform.translation.x = t_current.x();
  transform_msg.transform.translation.y = t_current.y();
  transform_msg.transform.translation.z = t_current.z();
  tf_broadcaster_.sendTransform(transform_msg);

  // This hacky thing is to make sure we can use the same rviz configuration no
  // matter the source of the data
  geometry_msgs::TransformStamped alias_transform_msg;
  alias_transform_msg.header.stamp = stamp;
  alias_transform_msg.header.frame_id = pointcloud_frame_;
  alias_transform_msg.child_frame_id = original_pointcloud_frame;
  alias_transform_msg.transform.translation.x = 0.0;
  alias_transform_msg.transform.translation.y = 0.0;
  alias_transform_msg.transform.translation.z = 0.0;
  alias_transform_msg.transform.rotation.x = 0.0;
  alias_transform_msg.transform.rotation.y = 0.0;
  alias_transform_msg.transform.rotation.z = 0.0;
  alias_transform_msg.transform.rotation.w = 1.0;
  tf_broadcaster_.sendTransform(alias_transform_msg);

  // publish odometry msg
  nav_msgs::Odometry odom_msg;
  odom_msg.header.stamp = stamp;
  odom_msg.header.frame_id = odom_frame_;
  odom_msg.child_frame_id = pointcloud_frame_;
  odom_msg.pose.pose.orientation.x = q_current.x();
  odom_msg.pose.pose.orientation.y = q_current.y();
  odom_msg.pose.pose.orientation.z = q_current.z();
  odom_msg.pose.pose.orientation.w = q_current.w();
  odom_msg.pose.pose.position.x = t_current.x();
  odom_msg.pose.pose.position.y = t_current.y();
  odom_msg.pose.pose.position.z = t_current.z();
  odom_publisher_.publish(odom_msg);

  // Publish trajectory msg
  geometry_msgs::PoseStamped pose_msg;
  pose_msg.pose = odom_msg.pose.pose;
  pose_msg.header = odom_msg.header;
  path_msg_.poses.push_back(pose_msg);
  traj_publisher_.publish(path_msg_);

  // Publish point cloud
  std_msgs::Header frame_header = lio_ekf_.lidar_header_;
  frame_header.frame_id = pointcloud_frame_;

  frame_publisher_.publish(*std::move(kiss_icp_ros::utils::EigenToPointCloud2(
      lio_ekf_.getFrame_w(), frame_header)));
  kpoints_publisher_.publish(*std::move(kiss_icp_ros::utils::EigenToPointCloud2(
      lio_ekf_.getKetPoints_w(), frame_header)));

  // Map is referenced to the odometry_frame
  std_msgs::Header local_map_header = lio_ekf_.lidar_header_;
  local_map_header.frame_id = odom_frame_;

  std::vector<Eigen::Vector3d> tmpmap = lio_ekf_.LocalMap();
  lio_ekf_.TransformPoints(tmp, tmpmap);
  map_publisher_.publish(*std::move(
      kiss_icp_ros::utils::EigenToPointCloud2(tmpmap, local_map_header)));
}

} // namespace lio_ekf
