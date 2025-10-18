#pragma once

#include "raw_gs_model.h"
#include "tensorboard_logger.h"
#include "file_utils.h"
#include "cv_utils.h"

class Pipeline
{
public:
    Pipeline()
    {
        std::random_device rd;
        rng.seed(rd());
        torch::manual_seed(42);
    }

    ~Pipeline() = default;

    void loadConfig(const YAML::Node &config, const std::string &workspace_dir, bool is_train);

    // supported mode: "auto", "only scalar", "scaler and image"
    void logResults(TensorDict &render_res, TensorDict &log_res, const Camera &cam, std::vector<std::string> names, std::string log_mode = "auto");

    void logScalars(TensorDict &scalars, int iter);

    void save(RawGaussianModel &model, std::vector<Camera> cams);

    // supported mode: "raw", "ges"
    void trainCams(RawGaussianModel &model, std::vector<Camera> cams, std::string train_mode = "raw")
    {
        if (train_mode == "raw")
            rawTrainCams(model, cams);
        else if (train_mode == "ges")
            gesTrainCams(model, cams);
    }

    void rawTrainCams(RawGaussianModel &model, std::vector<Camera> cams);

    void gesTrainCams(RawGaussianModel &model, std::vector<Camera> cams);

    void renderEvalImgs(RawGaussianModel &model, const std::vector<Camera> &cams, std::vector<std::string> eval_names);

    void logStartTime()
    {
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &start_time);
    }

    void logEndTime()
    {
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &end_time);
    }

    float getLogTime()
    {
        auto time = float(end_time.tv_nsec - start_time.tv_nsec) / 1000000 + float(end_time.tv_sec - start_time.tv_sec) * 1000;
        return time;
    }

    // 场景尺度，model densify会用到
    float scene_scale;
    // 采样使用的随机迭代器
    std::mt19937 rng;
    std::uniform_int_distribution<std::size_t> dist;
    int selected_cam_idx = -1;
    // 训练的iterations
    int curr_iter = 0;
    int max_iterations;
    bool enable_densify;
    bool eval_after_train, save_after_train;
    struct timespec start_time, end_time;

    YAML::Node config;
    YAML::Node weight_configs; // 权重的有关参数
    YAML::Node vis_configs;    // 可视化图片的有关参数
    // output setting
    TensorBoardLogger *tb_logger; // tensoboard的记录器
    std::string workspace_dir, model_path, log_path, eval_path, tb_path;
    int log_iter;

    // 保存mesh render的结果
    std::vector<torch::Tensor> mesh_rgbs;
    std::vector<torch::Tensor> mesh_depths;
};