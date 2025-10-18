#ifndef CV_UTILS_H
#define CV_UTILS_H

#include <torch/torch.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "ITMLib/Utils/ITMImageTypes.h"

// 把cv mat转换为torch tensor
torch::Tensor matToTensor(const cv::Mat &mat);

cv::Mat imreadRGB(const std::string &filename);

// 把float类型的颜色tensor转换成0-255的cv mat
cv::Mat tensorToImage(const torch::Tensor &t);

// 把float类型的颜色tensor转换成float的cv mat
cv::Mat tensorToDepth(const torch::Tensor &t);

cv::Mat imageToJetMat(const cv::Mat &depth_image, float min_depth, float max_depth);

// 把一个tensor使用Jet colormap可视化为cv Mat
cv::Mat tensorToJetMat(const torch::Tensor &tensor_input, float min, float max, bool set_zero);

// 比较两个颜色图像并拼接误差图
cv::Mat colorCompImg(const torch::Tensor &render_rgb,
                     const torch::Tensor &gt_rgb,
                     float min_error_thres,
                     float max_error_thres);

// 比较两个颜色图像并拼接误差图
cv::Mat depthCompImg(const torch::Tensor &render_depth,
                     const torch::Tensor &gt_depth,
                     float min_vis_depth, float max_vis_depth,
                     float min_error_thres, float max_error_thres);

torch::Tensor imageToTensor(const cv::Mat &image);

torch::Tensor depthToTensor(const cv::Mat &depth);

ITMUChar4Image *imageToITMUChar4Image(const cv::Mat &image, MemoryDeviceType type = MEMORYDEVICE_CPU);

ITMShortImage *depthToITMUShortImage(const cv::Mat &depth, MemoryDeviceType type = MEMORYDEVICE_CPU);

cv::Mat ITMUChar4ImageToImage(const ITMUChar4Image *itmu_image, MemoryDeviceType type = MEMORYDEVICE_CPU);

cv::Mat ITMUFloat4ImageToImage(const ITMFloat4Image *itmu_image, MemoryDeviceType type = MEMORYDEVICE_CPU);

torch::Tensor ITMUChar4ImageToTensor(ITMUChar4Image *itmu_image, MemoryDeviceType type = MEMORYDEVICE_CUDA);

torch::Tensor ITMUFloat4ImageToTensor(ITMFloat4Image *itmu_image, MemoryDeviceType type = MEMORYDEVICE_CUDA);

#endif