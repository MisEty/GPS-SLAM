#ifndef TENSOR_MATH_H
#define TENSOR_MATH_H

#include <torch/torch.h>
#include <tuple>
#include "../InfiniTAM/ORUtils/Matrix.h"

ORUtils::Matrix4<float> *tensorToInfiMatrix4(const torch::Tensor &tensor);

torch::Tensor infiMatrix4ToTensor(const ORUtils::Matrix4<float> &matrix);

/////////////////   Loss Functions    //////////////////
torch::Tensor l1(const torch::Tensor &rendered, const torch::Tensor &gt);

////////////////  Geo Transform    ////////////////////
torch::Tensor getR(const torch::Tensor &pose);

torch::Tensor getT(const torch::Tensor &pose);

torch::Tensor poseInv(const torch::Tensor &pose);

torch::Tensor verticesTransform(const torch::Tensor &vertex_tensor, const torch::Tensor transform);

torch::Tensor quatToRotMat(const torch::Tensor &quat);

std::tuple<torch::Tensor, torch::Tensor, float> autoScaleAndCenterPoses(const torch::Tensor &poses);

torch::Tensor rotationMatrix(const torch::Tensor &a, const torch::Tensor &b);

torch::Tensor rodriguesToRotation(const torch::Tensor &rodrigues);

torch::Tensor randomQuatTensor(long long n);

torch::Tensor computeQuat(const torch::Tensor &init_vec, const torch::Tensor &target_vec);

torch::Tensor quaternionFromAxisAngle(const torch::Tensor &axis, const torch::Tensor &angle);

torch::Tensor projectionMatrix(float zNear, float zFar, float fovX, float fovY, const torch::Device &device);

std::tuple<torch::Tensor, torch::Tensor> featureGradient(const torch::Tensor &img, bool normalize_gradient = true);

torch::Tensor computeVertexMap(const torch::Tensor &depth_map, const torch::Tensor intrinsics);

torch::Tensor computeNormalMap(const torch::Tensor &vertex_map);

float rotCompare(const torch::Tensor &prev_rot, const torch::Tensor &curr_rot);

float transCompare(const torch::Tensor &prev_trans, const torch::Tensor &curr_trans);
#endif