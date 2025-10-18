#include "slam_gs_model.h"

#define LOG_MODEL_TIME

void SLAMGaussianModel::addGaussians(const Camera &cam,
                                     TensorDict &frame_maps,
                                     const torch::Tensor &sample_mask,
                                     float new_gs_sample_ratio,
                                     int frame_num)
{
#ifdef LOG_MODEL_TIME
    struct timespec pixel_sample_start, pixel_sample_end, params_compute_end;
    clock_gettime(CLOCK_MONOTONIC, &pixel_sample_start);
#endif
    int height = cam.image.size(0);
    int width = cam.image.size(1);
    auto sample_mask_gpu = sample_mask.expand({height, width, 3}).to(device);

    auto valid_vertices = torch::masked_select(frame_maps["vertex_map"], sample_mask_gpu).reshape({-1, 3});
    auto valid_colors = torch::masked_select(cam.image.to(device), sample_mask_gpu).reshape({-1, 3});
    auto valid_normals = torch::masked_select(frame_maps["normal_map"], sample_mask_gpu).reshape({-1, 3});

    // 生成随机排列的索引
    auto valid_indices = torch::randperm(valid_vertices.size(0), valid_vertices.options().dtype(torch::kLong));
    int num_select = static_cast<int>(valid_vertices.size(0) * new_gs_sample_ratio);
    if (num_select > 0)
    {
        // 选择前num_select个索引
        auto selected_indices = valid_indices.slice(0, 0, num_select);
        // 选择对应的属性
        auto selected_vertices = valid_vertices.index_select(0, selected_indices);
        auto selected_colors = valid_colors.index_select(0, selected_indices);
        auto selected_normals = valid_normals.index_select(0, selected_indices);
#ifdef LOG_MODEL_TIME
        clock_gettime(CLOCK_MONOTONIC, &pixel_sample_end);
#endif
        // 生成高斯
        RawGaussianParams new_gs_params;
        new_gs_params.init(selected_vertices,
                           selected_colors,
                           selected_normals,
                           maxSH,
                           defaultOpacities,
                           maxInitScale,
                           minInitScale,
                            frame_num);
        new_gs_params.toGPU();
        opt_gs_params.add(new_gs_params);
#ifdef LOG_MODEL_TIME
        clock_gettime(CLOCK_MONOTONIC, &params_compute_end);
        printf("[MODEL TIME] pixel_sample: %f, params_compute: %f\n",
               calculateTimeInterval(pixel_sample_start, pixel_sample_end),
               calculateTimeInterval(pixel_sample_end, params_compute_end));
#endif
    }
}