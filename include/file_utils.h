#pragma once

#include <filesystem>
#include <iostream>
#include <system_error>
#include <fstream>
#include <nvml.h>

#include <Eigen/Dense>
#include <torch/torch.h>
#include <time.h>

#include "yaml-cpp/yaml.h"

///////////////// EIGEN LOAD, SAVE and TRANSFORM /////////////////
typedef Eigen::Vector3f Vector3;
typedef Eigen::Vector4f Vector4;
typedef Eigen::Matrix3f Matrix3;
typedef Eigen::Matrix4f Matrix4;
typedef Eigen::MatrixXf MatrixX;

MatrixX readMatrixFromTXT(const std::string &filename);

torch::Tensor matrixToTensor(const MatrixX &eigenMatrix);

//////////////////////  DATA SAVE AND LOAD ///////////////
bool createDirectory(const std::string &path, bool overwrite);

std::string idToFilename(int frame_id);

void createWorkSpace(const std::string &config_filename);

void saveCfgArgs(const std::string &file_name, int maxSH);

unsigned long long getGPUMemoryUsage(int gpu_id = 0);

double calculateTimeInterval(struct timespec start, struct timespec end);

void saveTensorTXT(const torch::Tensor &tensor, const std::string &filename);