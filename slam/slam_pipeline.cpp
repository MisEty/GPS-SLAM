#include "slam_pipeline.h"
#include "tensor_math.h"
#include <cstdio>

namespace fs = std::filesystem;

#define LOG_PIPELINE_TIME

bool compareLossInfo(const std::pair<std::string, std::vector<float>> &a,
                     const std::pair<std::string, std::vector<float>> &b)
{
    // 确保 vector 不为空
    if (a.second.empty() || b.second.empty())
    {
        return false; // 或者根据你的需求处理空 vector 的情况
    }
    return a.second[0] > b.second[0];
}

bool compareIdInfo(const std::pair<std::string, std::vector<float>> &a,
                   const std::pair<std::string, std::vector<float>> &b)
{
    // 确保 vector 不为空
    if (a.second.empty() || b.second.empty())
    {
        return false; // 或者根据你的需求处理空 vector 的情况
    }
    return a.second[1] < b.second[1];
}

class CompareSampleInfo
{
private:
    int curr_frame_id;
    float weight_intervel;
    float loss_thres;

public:
    CompareSampleInfo(int t, float weight, float loss) : curr_frame_id(t), weight_intervel(weight), loss_thres(loss) {}

    bool operator()(const std::pair<std::string, std::vector<float>> &a,
                    const std::pair<std::string, std::vector<float>> &b) const
    {
        float weight_intervel = 0.01;
        float loss_thres = 0.02;
        float sample_score_a = 1 - exp(weight_intervel * (a.second[1] - curr_frame_id)) + exp(a.second[0] / loss_thres - 1);
        float sample_score_b = 1 - exp(weight_intervel * (b.second[1] - curr_frame_id)) + exp(b.second[0] / loss_thres - 1);
        return sample_score_a > sample_score_b;
    }
};

void SLAMPipeline::SLAMTrainCams(SLAMGaussianModel &model, std::vector<Camera> &cams)
{
#ifdef LOG_PIPELINE_TIME
    struct timespec slam_start, slam_end, perFrame_start, perFrame_end,
        localFrameRaycast_start, localFrameRaycast_end,
        keyFrameRaycast_end, initNewGaussians_end, localOptimize_end, removeGaussian_end, checkError_end;
    double perFrame_total = 0.0, localFrameRaycast_total = 0.0, keyFrameRaycast_total = 0.0, initNewGaussians_total = 0.0, localOptimize_total = 0.0, removeGaussian_total = 0.0, checkError_total = 0.0;
    FILE *time_log_file = fopen((fs::path(workspace_dir) / "time_log.txt").c_str(), "w");
    if (time_log_file == nullptr)
    {
        perror("Failed to open file");
        return;
    }
    clock_gettime(CLOCK_MONOTONIC, &slam_start);
#endif
    int total_frame_num = cams.size();
    // TODO: 把3改成config里设置的gpu id
    for (int i = 0; i < total_frame_num; i++)
    {
#ifdef LOG_PIPELINE_TIME
        clock_gettime(CLOCK_MONOTONIC, &perFrame_start);
#endif
        curr_frame_id = i;
        // 1. 执行TSDF Fusion
        assert(curr_frame_id == tsdf_engine->currentFrameNo);
        tsdf_engine->ProcessFrame();

        // std::cout << "gt" << std::endl;
        // std::cout << cams[i].c2w << std::endl;
        torch::Tensor est_pose = infiMatrix4ToTensor(main_engine->GetTrackingState()->pose_d->GetInvM());
        cams[i].c2w_slam = est_pose;
        curr_cam = cams[i];
        curr_cam.toGPU();

        // std::cout << "slam" << std::endl;
        // std::cout << curr_cam.c2w_slam << std::endl;
        // 2.更新关键帧
        updateFrameList();
#ifdef LOG_PIPELINE_TIME
        clock_gettime(CLOCK_MONOTONIC, &perFrame_end);
        auto perFrame_time = calculateTimeInterval(perFrame_start, perFrame_end);
        perFrame_total += perFrame_time;
        printf("[PIPELINE TIME] cam id: %d, per frame fusion time: %f\n", curr_frame_id, perFrame_time);
#endif
        if (work_mode == "recon")
            continue;
        // 3. 每隔一定间隔进行高斯操作
        if (curr_frame_id % local_opt_interval == 0 && curr_frame_id > 0)
        {
#ifdef LOG_PIPELINE_TIME
            clock_gettime(CLOCK_MONOTONIC, &localFrameRaycast_start);
#endif
            // 更新所有local frame的raycast结果
            localFrameRaycast();
#ifdef LOG_PIPELINE_TIME
            clock_gettime(CLOCK_MONOTONIC, &localFrameRaycast_end);
#endif
            // 更新被选择的keyframe的raycast结果
            keyFrameRaycast(model);
#ifdef LOG_PIPELINE_TIME
            clock_gettime(CLOCK_MONOTONIC, &keyFrameRaycast_end);
#endif
            // 在当前帧添加新的可见高斯
            initNewGaussians(model, localframe_raycast_window.back());
#ifdef LOG_PIPELINE_TIME
            clock_gettime(CLOCK_MONOTONIC, &initNewGaussians_end);
#endif
            // 进行优化
            localOptimize(model);

#ifdef LOG_PIPELINE_TIME
            clock_gettime(CLOCK_MONOTONIC, &localOptimize_end);
#endif
            // 移除多余的高斯
            removeRedundantGs(model);
#ifdef LOG_PIPELINE_TIME
            clock_gettime(CLOCK_MONOTONIC, &removeGaussian_end);
#endif
            if (config["keyframe_sample_configs"]["sample_method"].as<std::string>() == "ours")
                checkKeyFrameError(model);
#ifdef LOG_PIPELINE_TIME
            clock_gettime(CLOCK_MONOTONIC, &checkError_end);
#endif
            // std::cout << "total gs num: " << model.getGaussianNum() << std::endl;
            // printf("gs scale mean/min/max: %f, %f, %f\n",
            //        std::get<0>(model.getRealScales().max(-1)).mean().item<float>(),
            //        std::get<0>(model.getRealScales().max(-1)).min().item<float>(),
            //        std::get<0>(model.getRealScales().max(-1)).max().item<float>());
            // printf("gs opac mean/min/max: %f, %f, %f\n",
            //        model.getRealOpacities().mean().item<float>(),
            //        model.getRealOpacities().min().item<float>(),
            //        model.getRealOpacities().max().item<float>());
#ifdef LOG_PIPELINE_TIME
            auto localFrameRaycast_time = calculateTimeInterval(localFrameRaycast_start, localFrameRaycast_end);
            auto keyFrameRaycast_time = calculateTimeInterval(localFrameRaycast_end, keyFrameRaycast_end);
            auto initNewGaussians_time = calculateTimeInterval(keyFrameRaycast_end, initNewGaussians_end);
            auto localOptimize_time = calculateTimeInterval(initNewGaussians_end, localOptimize_end);
            auto removeGaussian_time = calculateTimeInterval(localOptimize_end, removeGaussian_end);
            auto checkError_time = calculateTimeInterval(removeGaussian_end, checkError_end);

            localFrameRaycast_total += localFrameRaycast_time;
            keyFrameRaycast_total += keyFrameRaycast_time;
            initNewGaussians_total += initNewGaussians_time;
            localOptimize_total += localOptimize_time;
            removeGaussian_total += removeGaussian_time;
            checkError_total += checkError_time;

            printf("[PIPELINE TIME] localFrameRaycast time: %f, keyFrameRaycast time: %f, initNewGaussians time: %f, localOptimize time: %f, removeGaussian time: %f, checkError time: %f\n", localFrameRaycast_time, keyFrameRaycast_time, initNewGaussians_time, localOptimize_time, removeGaussian_time, checkError_time);
#endif
        }
    }
#ifdef LOG_PIPELINE_TIME
    clock_gettime(CLOCK_MONOTONIC, &slam_end);
    auto slam_total_time = calculateTimeInterval(slam_start, slam_end);
    printf("[PIPELINE AVG TIME] GS num: %d, per frame fusion time: %f, localFrameRaycast time: %f, keyFrameRaycast time: %f, initNewGaussians time: %f, localOptimize time: %f, FPS: %f\n", model.getGaussianNum(), perFrame_total / total_frame_num, localFrameRaycast_total / total_frame_num, keyFrameRaycast_total / total_frame_num, initNewGaussians_total / total_frame_num, localOptimize_total / total_frame_num, total_frame_num / (slam_total_time / 1000));
    fprintf(time_log_file, "[PIPELINE AVG TIME] GS num: %d, per frame fusion time: %f, localFrameRaycast time: %f, keyFrameRaycast time: %f, initNewGaussians time: %f, localOptimize time: %f, FPS: %f\n", model.getGaussianNum(), perFrame_total / total_frame_num, localFrameRaycast_total / total_frame_num, keyFrameRaycast_total / total_frame_num, initNewGaussians_total / total_frame_num, localOptimize_total / total_frame_num, total_frame_num / (slam_total_time / 1000));
    c10::cuda::CUDACachingAllocator::emptyCache();
    auto mem_curr = getGPUMemoryUsage(device_id);
    printf("GPU memory usage: %d MB\n", int(mem_curr));
    fprintf(time_log_file, "GPU memory usage: %d MB\n", int(mem_curr));
#endif
}

void SLAMPipeline::loadConfig(const YAML::Node &config, const std::string &workspace_dir, bool is_train)
{
    Pipeline::loadConfig(config, workspace_dir, is_train);
    new_gs_sample_ratio = config["new_gs_sample_ratio"].as<float>();
    keyframe_select_max = config["keyframe_select_max"].as<int>();
    localframe_cam_window_length = config["localframe_cam_window_length"].as<int>();
    localframe_cam_window_interval = config["localframe_cam_window_interval"].as<int>();
    local_opt_iters = config["local_opt_iters"].as<int>();
    local_opt_interval = config["local_opt_interval"].as<int>();
    color_error_thres = config["color_error_thres"].as<float>();
    keyframe_theta_thres = config["keyframe_theta_thres"].as<float>();
    keyframe_trans_thres = config["keyframe_trans_thres"].as<float>();
    log_slam_state = config["log_slam_state"].as<bool>();
    if (is_train)
    {
        createDirectory(workspace_dir + "/" + config["TSDF"]["saved_images"].as<std::string>(), true);
        createDirectory(workspace_dir + "/before_opt", true);
    }
}

void SLAMPipeline::localOptimize(SLAMGaussianModel &model)
{
    struct timespec init_start, init_end;
    clock_gettime(CLOCK_MONOTONIC, &init_start);

    model.setParamsDevice(model.device);
    model.setParamsRequireGrad();
    model.initOptimizers(-1, scene_scale);
    RandomSelector data_loader(opt_cam_list);
#ifndef LOG_PIPELINE_TIME
    // ProgressBar
    show_console_cursor(false);
    ProgressBar bar{
        option::BarWidth{50},
        option::Start{"["},
        option::Fill{"="},
        option::Lead{">"},
        option::Remainder{" "},
        option::End{"]"},
        option::ForegroundColor{Color::green},
        option::ShowPercentage{true},
        option::ShowElapsedTime{true},
        option::ShowRemainingTime{true},
        option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}};
#endif
    clock_gettime(CLOCK_MONOTONIC, &init_end);
    float iter_time_total = 0;
    for (int iter = 0; iter < local_opt_iters; iter++)
    {
        struct timespec opt_start, opt_end, preprocess_end, forward_end, backward_end, log_end;
        clock_gettime(CLOCK_MONOTONIC, &opt_start);

        Camera cam;
        int cam_idx;
        // cam = opt_cam_list.back();
        // cam_idx = opt_cam_list.size() - 1;
        // 随机选择优化的相机
        auto sample = data_loader.getNext();
        cam = sample.value;
        cam_idx = sample.originalIndex;
        if ((iter == 0 || iter == local_opt_iters - 1) && log_slam_state)
        {
            cam = curr_cam;
            cam_idx = localframe_raycast_window.size() - 1;
        }

        // 始终采用全部SH
        model.updateSH(-1);
        // 渲染, 始终不启用densify
        auto raycast_depth = opt_raycast_list[cam_idx]["depth_map"];
        auto raycast_color = opt_raycast_list[cam_idx]["color_map"];
        clock_gettime(CLOCK_MONOTONIC, &preprocess_end);
        auto render_res = model.forward(cam, raycast_depth, raycast_color);
        TensorDict loss = model.computeLoss(render_res, cam, weight_configs);
        // std::cout << cam.getFrameID() << "\t" << loss["total"].item() << std::endl;
        clock_gettime(CLOCK_MONOTONIC, &forward_end);
        loss["total"].backward();
        clock_gettime(CLOCK_MONOTONIC, &backward_end);
        model.optimizersStep();
        model.optimizersZeroGrad();
        clock_gettime(CLOCK_MONOTONIC, &opt_end);
        iter_time_total += calculateTimeInterval(opt_start, opt_end);
        // printf("[PIPELINE TIME] preprocess time: %f, forward time: %f, backward time: %f, optimizer time: %f\n",
        //        calculateTimeInterval(opt_start, preprocess_end),
        //        calculateTimeInterval(preprocess_end, forward_end),
        //        calculateTimeInterval(forward_end, backward_end),
        //        calculateTimeInterval(backward_end, opt_end));
        if (log_slam_state)
        {
            // tensorboard记录结果
            TensorDict log_info = loss;
            log_info["opt_state/opt_gs_num"] = torch::tensor(model.getGaussianNum()).to(torch::kInt32);
            log_info["opt_state/iter_time"] = torch::tensor(calculateTimeInterval(opt_start, opt_end)).to(torch::kFloat32);
            log_info["opt_state/opacity_mean"] = model.getRealOpacities().mean();
            logResults(render_res, log_info, cam, {"rgb", "alpha"}, "only scalar");
            clock_gettime(CLOCK_MONOTONIC, &log_end);
            // printf("[PIPELINE TIME] preprocess time: %f, forward time: %f, backward time: %f, optimizer time: %f, log time: %f\n",
            //        calculateTimeInterval(opt_start, preprocess_end),
            //        calculateTimeInterval(preprocess_end, forward_end),
            //        calculateTimeInterval(forward_end, backward_end),
            //        calculateTimeInterval(backward_end, opt_end),
            //        calculateTimeInterval(opt_end, log_end));
            // std::cout << raycast_depth.mean().item() << "\t" << raycast_color.mean().item() << std::endl;
            if ((iter == 0 || iter == local_opt_iters - 1))
                logResults(render_res, log_info, cam, {"rgb", "alpha"}, "scaler and image");
            else
                logResults(render_res, log_info, cam, {"rgb", "alpha"}, "only scalar");
        }
#ifndef LOG_PIPELINE_TIME
        // 更新progress bar的进度
        bar.set_option(option::PostfixText{"Frame: " + std::to_string(curr_frame_id) + " Iter: " + std::to_string(iter) + "/" + std::to_string(local_opt_iters) + " Loss: " + std::to_string(loss["total"].item<float>())});
        bar.set_progress(100 * iter / local_opt_iters);
#endif
        curr_iter++;
    }
    printf("per iter time: %f, opt setup time: %f\n", iter_time_total / local_opt_iters, calculateTimeInterval(init_start, init_end));
}

void SLAMPipeline::checkKeyFrameError(SLAMGaussianModel &model)
{
    int localframe_cam_window_size = localframe_cam_window.size();
    std::vector<Camera> opt_key_cam_list(opt_cam_list.begin() + localframe_cam_window_size, opt_cam_list.end());
    std::vector<TensorDict> opt_key_raycast_list(opt_raycast_list.begin() + localframe_cam_window_size, opt_raycast_list.end());
    // 更新keyframe的loss dict
    for (int key_cam_idx = 0; key_cam_idx < opt_key_cam_list.size(); key_cam_idx++)
    {
        auto opt_key_cam = opt_key_cam_list[key_cam_idx];
        auto opt_key_raycast_depth = opt_key_raycast_list[key_cam_idx]["depth_map"];
        auto opt_key_raycast_color = opt_key_raycast_list[key_cam_idx]["color_map"];
        auto render_res = model.forward(opt_key_cam, opt_key_raycast_depth, opt_key_raycast_color);
        TensorDict loss = model.computeLoss(render_res, opt_key_cam, weight_configs, opt_key_raycast_depth > 0);
        float confidence_mean = opt_key_raycast_list[key_cam_idx]["confidence_map"].mean().item<float>();
        float opt_count = keyframe_loss_dict[opt_key_cam.getFrameID()][3];
        if (loss["total"].item<float>() > config["keyframe_sample_configs"]["loss_thres"].as<float>())
            opt_count += 1;
        // loss, frame_id, confidence, opt_count
        std::vector<float> loss_info = {loss["total"].item<float>(),
                                        float(curr_frame_id),
                                        confidence_mean,
                                        opt_count};
        keyframe_loss_dict[opt_key_cam.getFrameID()] = loss_info;
    }
}

void SLAMPipeline::updateFrameList()
{
    if (curr_frame_id == 0)
        return;
    // 维护一个固定长度的局部帧窗口用于局部优化
    if (curr_frame_id % localframe_cam_window_interval == 0)
    {
        localframe_cam_window.push_back(curr_cam);
        if (localframe_cam_window.size() == localframe_cam_window_length + 1)
            localframe_cam_window.pop_front();
    }

    // 判断是否是关键帧
    is_keyframe = false;
    if (keyframe_cam_list.empty())
    {
        is_keyframe = true;
    }
    else
    {
        auto last_keyframe_cam = keyframe_cam_list.back();
        auto prev_rot = getR(last_keyframe_cam.c2w_slam);
        auto prev_trans = getT(last_keyframe_cam.c2w_slam);
        auto curr_rot = getR(curr_cam.c2w_slam);
        auto curr_trans = getT(curr_cam.c2w_slam);
        float theta_diff = rotCompare(prev_rot, curr_rot);
        float trans_diff = transCompare(prev_trans, curr_trans);
        // std::cout << "theta diff: " << theta_diff << "\t" << "trans diff: " << trans_diff << std::endl;
        if (theta_diff > keyframe_theta_thres || trans_diff > keyframe_trans_thres)
            is_keyframe = true;
    }
    if (is_keyframe)
    {
        std::cout << "add keyframe!" << std::endl;
        keyframe_cam_list.push_back(curr_cam);
        keyframe_cam_dict[curr_cam.getFrameID()] = curr_cam;
        keyframe_loss_dict[curr_cam.getFrameID()] = {0.1, float(curr_frame_id), 0, 0, 0};
    }
    // std::cout << localframe_raycast_window[0]["color_map"][100][200] << std::endl;
    // std::cout << localframe_raycast_window[0]["vertex_map"][100][200] << std::endl;
    return;
}

TensorDict SLAMPipeline::runRaycastByCam(const Camera &cam, bool use_cam_depth)
{
    TensorDict raycast_res;
    ORUtils::SE3Pose cam_pose;
    ITMIntrinsics cam_intrinc;
    if (dynamic_cast<ITMBasicEngine<ITMVoxel, ITMVoxelIndex> *>(main_engine)->camPoses.size() > 0 && cam.id >= 0)
    {
        int cam_id = cam.id;
        cam_pose = dynamic_cast<ITMBasicEngine<ITMVoxel, ITMVoxelIndex> *>(main_engine)->camPoses[cam_id];
        cam_intrinc = dynamic_cast<ITMBasicEngine<ITMVoxel, ITMVoxelIndex> *>(main_engine)->camIntrincs[cam_id];
    }
    else
    {
        cam_pose.SetInvM(*tensorToInfiMatrix4(cam.c2w));
        cam_intrinc.SetFrom(cam.width, cam.height, cam.fx, cam.fy, cam.cx, cam.cy);
    }

    dynamic_cast<ITMBasicEngine<ITMVoxel, ITMVoxelIndex> *>(main_engine)->runRaycast(&cam_pose, &cam_intrinc);
    // get raycast res
    raycast_color_raw_temp = dynamic_cast<ITMBasicEngine<ITMVoxel, ITMVoxelIndex> *>(main_engine)->GetFreeImage();
    raycast_vertex_raw_temp = dynamic_cast<ITMBasicEngine<ITMVoxel, ITMVoxelIndex> *>(main_engine)->GetFreeVertex();
    // clock_gettime(CLOCK_MONOTONIC, &raycast_end);

    // convert to tensor
    torch::Tensor color_tensor = ITMUChar4ImageToTensor(raycast_color_raw_temp);
    // clock_gettime(CLOCK_MONOTONIC, &convert_end);

    torch::Tensor vertex_conf_tensor = ITMUFloat4ImageToTensor(raycast_vertex_raw_temp);
    torch::Tensor vertex_tensor = vertex_conf_tensor.slice(2, 0, 3).contiguous() * voxel_size;
    torch::Tensor conf_tensor = vertex_conf_tensor.slice(2, 3, 4).contiguous();
    raycast_res["color_map"] = color_tensor;
    raycast_res["vertex_map"] = vertex_tensor;
    raycast_res["confidence_map"] = conf_tensor;

    if (!use_cam_depth)
    {
        torch::Tensor w2c = poseInv(cam.c2w).to(device);
        torch::Tensor transformed_vertices = verticesTransform(vertex_tensor, w2c);
        raycast_res["depth_map"] = transformed_vertices.index({torch::indexing::Slice(), torch::indexing::Slice(), 2}).unsqueeze(-1).contiguous();
        torch::Tensor invalid_vertex_mask = (vertex_tensor.sum(2) == 0).unsqueeze(-1);
        raycast_res["depth_map"].masked_fill_(invalid_vertex_mask, 0);
    }

    else
    {
        raycast_res["depth_map"] = cam.depth.contiguous().to(device);
    }
    // clock_gettime(CLOCK_MONOTONIC, &depth_end);
    // printf("raycast time: %f, convert time: %f, depth time: %f\n",
    //        calculateTimeInterval(raycast_start, raycast_end),
    //        calculateTimeInterval(raycast_end, convert_end),
    //        calculateTimeInterval(convert_end, depth_end));
    return raycast_res;
}

void SLAMPipeline::localFrameRaycast()
{
    // 更新raycast results
    float color_error_max = vis_configs["color_error_max"].as<float>();
    float depth_vis_max = vis_configs["depth_vis_max"].as<float>();
    float depth_error_max = vis_configs["depth_error_max"].as<float>();

    localframe_raycast_window.clear();
    for (auto &localframe_cam : localframe_cam_window)
    {
        TensorDict localframe_raycast = runRaycastByCam(localframe_cam, false);
        localframe_raycast_window.push_back(localframe_raycast);
#ifndef LOG_PIPELINE_TIME
        // save raycast result
        // save color
        torch::Tensor raycast_rgb = torch::clamp(localframe_raycast["color_map"], 0, 1);
        cv::Mat comp_image = colorCompImg(raycast_rgb, localframe_cam.image, 0, color_error_max);
        cv::imwrite((fs::path(workspace_dir) /
                     config["TSDF"]["saved_images"].as<std::string>() /
                     ("frame" + curr_cam.getFrameID() + "_" + localframe_cam.getFrameID() + ".color.jpg"))
                        .string(),
                    comp_image);
        // save depth
        cv::Mat comp_depth = depthCompImg(localframe_raycast["depth_map"], localframe_cam.depth, 0, depth_vis_max, 0, depth_error_max);
        cv::imwrite((fs::path(workspace_dir) /
                     config["TSDF"]["saved_images"].as<std::string>() /
                     ("frame" + curr_cam.getFrameID() + "_" + localframe_cam.getFrameID() + ".depth.jpg"))
                        .string(),
                    comp_depth);
#endif
    }
}

void SLAMPipeline::initNewGaussians(SLAMGaussianModel &model, TensorDict &raycast_maps)
{
#ifdef LOG_PIPELINE_TIME
    struct timespec depth_check_start, depth_check_end, render_end, color_check_end, normal_compute_end, add_gaussian_end;
    clock_gettime(CLOCK_MONOTONIC, &depth_check_start);
#endif
    torch::NoGradGuard noGrad;
    float depth_range_max = vis_configs["depth_vis_max"].as<float>();
    float depth_range_min = vis_configs["depth_vis_min"].as<float>();
    float empty_alpha_max = vis_configs["alpha_vis_max"].as<float>();
    auto raycast_depth = raycast_maps["depth_map"];
    auto raycast_color = raycast_maps["color_map"];
    auto raycast_vertex = raycast_maps["vertex_map"];
    int frame_num = local_opt_interval;

    torch::Tensor sample_mask;
    auto valid_depth_mask = ((raycast_depth > depth_range_min) & (raycast_depth < depth_range_max)).to(model.device);
    torch::Tensor valid_vertex_mask = (raycast_vertex.sum(2) == 0).unsqueeze(-1);
    valid_depth_mask = valid_depth_mask & (~valid_vertex_mask);

    // std::cout << "valid depth ratio: " << (valid_depth_mask.sum() / (curr_cam.width * curr_cam.height)).item() << std::endl;
#ifdef LOG_PIPELINE_TIME
    clock_gettime(CLOCK_MONOTONIC, &depth_check_end);
#endif
    if (model.getGaussianNum() == 0)
    {
#ifdef LOG_PIPELINE_TIME
        clock_gettime(CLOCK_MONOTONIC, &render_end);
#endif
        // 第一帧没有高斯，只需要根据mesh判断颜色误差
        auto color_error = torch::mean(torch::abs(raycast_color - curr_cam.image.to(model.device)), -1, true);
        auto color_error_mask = (color_error > color_error_thres) & valid_depth_mask;
        sample_mask = color_error_mask;
        frame_num += 1;
    }
    else
    {
        // 之后的帧根据alpha/color/depth误差采样
        auto render_res = model.forward(curr_cam, raycast_depth, raycast_color);
        auto render_alpha = render_res["alpha"];
#ifdef LOG_PIPELINE_TIME
        clock_gettime(CLOCK_MONOTONIC, &render_end);
#endif
        auto color_error = torch::mean(torch::abs(render_res["rgb"] - curr_cam.image.to(model.device)), -1, true);
        auto color_error_mask = (color_error > color_error_thres) & valid_depth_mask & (render_alpha < empty_alpha_max);
        sample_mask = color_error_mask;
#ifdef LOG_PIPELINE_TIME
        clock_gettime(CLOCK_MONOTONIC, &color_check_end);
#endif
#ifndef LOG_PIPELINE_TIME
        // save compare image
        torch::Tensor rendered_rgb = torch::clamp(render_res["rgb"], 0, 1);
        float color_error_max = vis_configs["color_error_max"].as<float>();
        cv::Mat comp_image = colorCompImg(rendered_rgb, curr_cam.image, 0, color_error_max);
        cv::imwrite((fs::path(workspace_dir) / "before_opt" / ("frame" + curr_cam.getFrameID() + ".jpg")).string(), comp_image);
        // std::cout << "color error all: " << color_error.mean().item() << std::endl;
        // std::cout << "color error gs: " << color_error.masked_select(render_alpha > 0).mean().item() << std::endl;
        // std::cout << "color error tsdf: " << color_error.masked_select(render_alpha == 0).mean().item() << std::endl;
        // std::cout << "color error pixel num: " << color_error_mask.sum().item() << std::endl;
#endif
    }
    raycast_maps["normal_map"] = computeNormalMap(raycast_maps["vertex_map"]);
#ifdef LOG_PIPELINE_TIME
    clock_gettime(CLOCK_MONOTONIC, &normal_compute_end);
#endif

    model.addGaussians(curr_cam, raycast_maps, sample_mask, new_gs_sample_ratio, frame_num);
#ifdef LOG_PIPELINE_TIME
    clock_gettime(CLOCK_MONOTONIC, &add_gaussian_end);
    printf("[PIPELINE TIME] depth_check: %f, render: %f, color_check: %f, normal_compute: %f, add gaussian: %f\n",
           calculateTimeInterval(depth_check_start, depth_check_end),
           calculateTimeInterval(depth_check_end, render_end),
           calculateTimeInterval(render_end, color_check_end),
           calculateTimeInterval(color_check_end, normal_compute_end),
           calculateTimeInterval(normal_compute_end, add_gaussian_end));
#endif
}

void SLAMPipeline::keyFrameRaycast(SLAMGaussianModel &model)
{
    torch::NoGradGuard noGrad;
    opt_cam_list.clear();
    opt_raycast_list.clear();
    // 1. 从localframe window中添加
    opt_cam_list.assign(localframe_cam_window.begin(), localframe_cam_window.end());
    opt_raycast_list.assign(localframe_raycast_window.begin(), localframe_raycast_window.end());

    // 2. 从keyframe中添加
    if (config["keyframe_sample_configs"]["sample_method"].as<std::string>() == "random")
    {
        // 随机选取历史关键帧
        int global_select_num = std::min(keyframe_select_max, int(keyframe_cam_list.size()));
        RandomSelector data_loader(keyframe_cam_list);
        for (int i = 0; i < global_select_num; i++)
        {
            auto sample = data_loader.getNext();
            auto cam = sample.value;
            auto cam_idx = sample.originalIndex;
            auto keyframe_raycast_res = runRaycastByCam(cam, false);
            opt_cam_list.emplace_back(cam);
            opt_raycast_list.emplace_back(keyframe_raycast_res);
        }
    }

    // 4. 输出优化的相机id
    std::cout << "opt cam id: ";
    for (auto opt_cam : opt_cam_list)
    {
        std::cout << opt_cam.getFrameID() << " ";
    }
    std::cout << std::endl;
}

// 移除场景中一些冗余的高斯，包括不透明度较低/尺度太小的高斯
void SLAMPipeline::removeRedundantGs(SLAMGaussianModel &model)
{
    if (model.getGaussianNum() == 0)
        return;
    float min_opac = config["remove_configs"]["low_opac_thres"].as<float>();
    float min_scale = config["remove_configs"]["small_scale_thres"].as<float>();
    float max_scale = config["remove_configs"]["large_scale_thres"].as<float>();

    auto small_scale_gs_mask = (std::get<0>(model.getRealScales().max(-1)) < min_scale);
    auto max_scale_gs_mask = (std::get<0>(model.getRealScales().max(-1)) > max_scale);
    auto low_opac_gs_mask = model.getRealOpacities().squeeze() < min_opac;

    auto remove_mask = small_scale_gs_mask | max_scale_gs_mask | low_opac_gs_mask;
    printf("small scale gs: %d, large scale gs: %d, low opac gs: %d, total remove gs: %d\n",
           small_scale_gs_mask.sum().item<int>(),
           max_scale_gs_mask.sum().item<int>(),
           low_opac_gs_mask.sum().item<int>(),
           remove_mask.sum().item<int>());
    if (remove_mask.sum().item<int>() > 0)
    {
        model.prunePoints(remove_mask);
    }
}

void SLAMPipeline::renderEvalImgs(SLAMGaussianModel &model, const std::vector<Camera> &cams, std::vector<std::string> names)
{
    torch::NoGradGuard noGrad;
    // ProgressBar
    show_console_cursor(false);
    ProgressBar bar{
        option::BarWidth{50},
        option::Start{"["},
        option::Fill{"="},
        option::Lead{">"},
        option::Remainder{" "},
        option::End{"]"},
        option::PostfixText{"Eval cameras"},
        option::ForegroundColor{Color::green},
        option::ShowPercentage{true},
        option::ShowElapsedTime{true},
        option::ShowRemainingTime{true},
        option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}};

    float color_error_max = vis_configs["color_error_max"].as<float>();
    float depth_error_max = vis_configs["depth_error_max"].as<float>();
    float depth_vis_max = vis_configs["depth_vis_max"].as<float>();
    fs::create_directories(fs::path(eval_path) / "gt");
    fs::create_directories(fs::path(eval_path) / "render");
    fs::create_directories(fs::path(eval_path) / "comp");
    fs::create_directories(fs::path(eval_path) / "raycast_depth");
    fs::create_directories(fs::path(eval_path) / "raycast_color");
    // fs::create_directories(fs::path(eval_path) / "pose");

    int read_count = 0;
    for (Camera cam : cams)
    {
        TensorDict render_res;
        std::string frame_file_name = "frame" + cam.getFrameID() + "_iter" + std::to_string(curr_iter);

        auto raycast_res = runRaycastByCam(cam, false);
        torch::Tensor raycast_color = raycast_res["color_map"];
        torch::Tensor raycast_depth = raycast_res["depth_map"];
        // save raycast image
        cv::Mat raycast_color_img = tensorToImage(raycast_color);
        cv::imwrite((fs::path(eval_path) / "raycast_color" / ("frame" + cam.getFrameID() + ".color.jpg")).string(),
                    raycast_color_img);
        torch::Tensor gt_rgb = cam.image;
        cv::Mat comp_image = colorCompImg(raycast_color, gt_rgb, 0, color_error_max);
        cv::imwrite((fs::path(eval_path) / "raycast_color" / ("frame" + cam.getFrameID() + "_comp.color.jpg")).string(),
                    comp_image);
        // save raycast depth
        cv::Mat raycast_depth_img = tensorToDepth(raycast_depth);
        cv::imwrite((fs::path(eval_path) / "raycast_depth" / ("frame" + cam.getFrameID() + ".depth.png")).string(),
                    raycast_depth_img);
        torch::Tensor gt_depth = cam.depth;
        cv::Mat comp_depth = depthCompImg(raycast_depth, gt_depth, 0, depth_vis_max, 0, depth_error_max);
        cv::imwrite((fs::path(eval_path) / "raycast_depth" / ("frame" + cam.getFrameID() + "_comp.depth.png")).string(), comp_depth);

        if (model.getGaussianNum() > 0)
        {
            render_res = model.forward(cam, raycast_depth, raycast_color);
            for (std::string name : names)
            {
                if (name == "rgb")
                {
                    torch::Tensor rendered_rgb = torch::clamp(render_res["rgb"], 0, 1);
                    torch::Tensor gt_rgb = cam.image;

                    // save gt image
                    cv::Mat gt_img = tensorToImage(gt_rgb);
                    cv::imwrite((fs::path(eval_path) / "gt" / (frame_file_name + ".color.jpg")).string(),
                                gt_img);
                    // save render image
                    cv::Mat rendered_img = tensorToImage(rendered_rgb);
                    cv::imwrite((fs::path(eval_path) / "render" / (frame_file_name + ".color.jpg")).string(),
                                rendered_img);
                    // save compare image
                    cv::Mat comp_image = colorCompImg(rendered_rgb, gt_rgb, 0, color_error_max);
                    cv::imwrite((fs::path(eval_path) / "comp" / (frame_file_name + ".color.jpg")).string(),
                                comp_image);
                }

                if (name == "alpha")
                {
                    torch::Tensor rendered_alpha = render_res["alpha"];
                    cv::Mat alpha_image = tensorToJetMat(rendered_alpha, 0, 1, false);
                    cv::imwrite((fs::path(eval_path) / "render" / (frame_file_name + ".alpha.jpg")).string(),
                                alpha_image);
                }

                if (name == "depth")
                {
                    torch::Tensor rendered_depth = render_res["depth"];
                    cv::Mat rendered_img = tensorToJetMat(rendered_depth, 0, depth_vis_max, true);
                    cv::imwrite((fs::path(eval_path) / "render" / (frame_file_name + ".depth.jpg")).string(), rendered_img);
                    if (cam.has_depth)
                    {
                        torch::Tensor gt_depth = cam.depth;
                        // save compare image
                        cv::Mat comp_depth = depthCompImg(rendered_depth, gt_depth, 0, depth_vis_max, 0, depth_error_max);
                        cv::imwrite((fs::path(eval_path) / "comp" / (frame_file_name + ".depth.jpg")).string(),
                                    comp_depth);
                    }
                }
            }
        }
        read_count++;
        bar.set_option(option::PostfixText{"Eval camera: " + frame_file_name});
        bar.set_progress(100 * read_count / cams.size());
    }
    show_console_cursor(true);
}