#include "pipeline.h"

namespace fs = std::filesystem;

void Pipeline::loadConfig(const YAML::Node &config, const std::string &workspace_dir, bool is_train)
{
    // std::cout << "loading training pipeline parameters...";
    this->config = config;
    max_iterations = config["max_iterations"].as<int>();
    selected_cam_idx = config["selected_cam_idx"].as<int>();
    enable_densify = config["enable_densify"].as<bool>();
    eval_after_train = config["eval_after_train"].as<bool>();
    save_after_train = config["save_after_train"].as<bool>();

    // output settings
    log_iter = config["log_iter"].as<int>();
    if (log_iter == -1)
        log_iter = max_iterations - 1;
    this->workspace_dir = workspace_dir;
    model_path = workspace_dir + config["model_path"].as<std::string>();
    log_path = workspace_dir + config["log_path"].as<std::string>();
    eval_path = workspace_dir + config["eval_path"].as<std::string>();

    tb_path = workspace_dir + "/tensorboard";
    weight_configs = config["weight_configs"];
    vis_configs = config["vis_configs"];
    if (is_train)
    {
        createDirectory(model_path, true);
        createDirectory(log_path, true);
        createDirectory(eval_path, true);
        createDirectory(tb_path, true);
        GOOGLE_PROTOBUF_VERIFY_VERSION;
        tb_logger = new TensorBoardLogger((fs::path(tb_path) / "tfevents.pb").string());
    }
}

void Pipeline::save(RawGaussianModel &model, std::vector<Camera> cams)
{
    if (model.getGaussianNum() == 0)
    {
        std::cout << "empty model" << std::endl;
        return;
    }
    std::cout << "save model start" << std::endl;
    std::string save_path = model_path;
    createDirectory(fs::path(save_path) / "point_cloud", true);

    saveCfgArgs((fs::path(save_path) / "cfg_args").string(), model.getMaxSH());
    model.saveParamsPly((fs::path(save_path) / "point_cloud" / "point_cloud.ply").string());
    model.saveParamsTensor((fs::path(save_path) / "model.pt").string());
    saveCameras(cams,
                (fs::path(save_path) / "cameras.json").string());
}

// 根据curr iter保存结果
void Pipeline::logResults(TensorDict &render_res, TensorDict &log_res, const Camera &cam, std::vector<std::string> names, std::string log_mode)
{
    torch::NoGradGuard noGrad;
    logScalars(log_res, curr_iter);
    logScalars(render_res, curr_iter);

    float color_error_max = vis_configs["color_error_max"].as<float>();
    float depth_error_max = vis_configs["depth_error_max"].as<float>();
    float depth_vis_max = vis_configs["depth_vis_max"].as<float>();
    float depth_vis_min = vis_configs["depth_vis_min"].as<float>();
    float alpha_vis_max = vis_configs["alpha_vis_max"].as<float>();
    bool log_image = false;
    if (log_mode == "only scalar")
        return;
    else if (log_mode == "auto")
        log_image = curr_iter % log_iter == 0 || curr_iter + 1 == max_iterations;
    else if (log_mode == "scaler and image")
        log_image = true;
    else
        std::cout << "UNSUPPORT LOG MODE!!!!!!!!!!!!" << std::endl;
    if (log_image)
    {
        std::cout << "log results, curr iteration: " << curr_iter << std::endl;
        for (std::string name : names)
        {
            std::string frame_file_name = "frame" + cam.getFrameID() + "_iter" + std::to_string(curr_iter);
            if (name == "rgb")
            {
                torch::Tensor rendered_rgb = torch::clamp(render_res["rgb"], 0, 1);
                torch::Tensor gt_rgb = cam.image;
                cv::Mat rgb_image = colorCompImg(rendered_rgb, gt_rgb, 0, color_error_max);
                cv::imwrite((fs::path(log_path) / (frame_file_name + ".color.jpg")).string(),
                            rgb_image);
            }
            if (name == "alpha")
            {
                torch::Tensor rendered_alpha = render_res["alpha"];
                std::cout << rendered_alpha.mean() << std::endl;
                std::cout << rendered_alpha.max() << std::endl;
                std::cout << rendered_alpha.min() << std::endl;

                cv::Mat alpha_image = tensorToJetMat(rendered_alpha, 0, alpha_vis_max, false);
                cv::imwrite((fs::path(log_path) /
                             (frame_file_name + ".alpha.jpg"))
                                .string(),
                            alpha_image);
            }
            if (name == "depth")
            {
                torch::Tensor rendered_depth = render_res["depth"];
                if (cam.has_depth)
                {
                    torch::Tensor gt_depth = cam.depth;
                    cv::Mat depth_image = depthCompImg(rendered_depth, gt_depth, depth_vis_min, depth_vis_max, 0, depth_error_max);
                    cv::imwrite((fs::path(log_path) /
                                 (frame_file_name + ".depth.jpg"))
                                    .string(),
                                depth_image);
                }
                else
                {
                    cv::Mat depth_image = tensorToJetMat(rendered_depth, depth_vis_min, depth_vis_max, true);
                    cv::imwrite((fs::path(log_path) /
                                 (frame_file_name + ".depth.jpg"))
                                    .string(),
                                depth_image);
                }
            }
        }
    }
}

// 记录Dict中的标量到tensorboard里
void Pipeline::logScalars(TensorDict &scalars, int iter)
{
    torch::NoGradGuard noGrad;
    for (auto &pair : scalars)
    {
        std::string key = pair.first;
        torch::Tensor tensor = pair.second;
        if (tensor.numel() == 1)
        {
            auto dtype = tensor.dtype();
            if (dtype == torch::kFloat32)
                tb_logger->add_scalar(key, iter, tensor.item<float>());
            else if (dtype == torch::kInt32)
                tb_logger->add_scalar(key, iter, float(tensor.item<int>()));
            else
                std::cout << key << " unsupported data type!" << dtype << std::endl;
        }
        // else
        // {
        //     std::cout << key << " tensor numel is not 1!" << std::endl;
        // }
    }
}

// 按照原始3dgs的pipeline进行训练
void Pipeline::rawTrainCams(RawGaussianModel &model, std::vector<Camera> cams)
{
    assert(model.getRenderMethod() == "raw");
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

    dist = std::uniform_int_distribution<std::size_t>(0, cams.size() - 1);
    model.setParamsDevice(model.device);
    model.setParamsRequireGrad();
    model.initOptimizers(max_iterations, scene_scale);
    RandomSelector data_loader(cams);
    // 开始优化
    for (; curr_iter < max_iterations; curr_iter++)
    {
        Camera cam;
        int cam_idx;
        // 随机选择优化的相机
        if (selected_cam_idx == -1)
        {
            auto sample = data_loader.getNext();
            cam = sample.value;
            cam_idx = sample.originalIndex;
        }
        else
        {
            cam = cams[selected_cam_idx];
            cam_idx = selected_cam_idx;
        }
        // 根据iter逐渐提升SH
        model.updateSH(curr_iter);
        // 渲染
        logStartTime();
        auto render_res = model.forward(cam);
        if (enable_densify)
        {
            render_res["means2d"].retain_grad();
        }
        TensorDict loss = model.computeLoss(render_res, cam, weight_configs);
        loss["total"].backward();
        logEndTime();

        model.optimizersStep();
        model.optimizersZeroGrad();
        if (enable_densify)
            model.stepPostBackward(render_res, cam, scene_scale, curr_iter);

        model.schedulersStep();
        // tensorboard记录结果
        TensorDict log_info = loss;
        auto duration = getLogTime();
        log_info["opt_state/opt_gs_num"] = torch::tensor(model.getGaussianNum()).to(torch::kInt32);
        log_info["opt_state/iter_time"] = torch::tensor(duration).to(torch::kFloat32);
        log_info["opt_state/opacity_mean"] = model.getRealOpacities().mean();
        logResults(render_res, log_info, cam, {"rgb"});

        // 更新progress bar的进度
        bar.set_option(option::PostfixText{"Training iter: " + std::to_string(curr_iter) + "/" + std::to_string(max_iterations) + " Loss: " + std::to_string(loss["total"].item<float>())});
        bar.set_progress(100 * curr_iter / max_iterations);
    }
}

// 按照ges-gs的pipeline进行训练
void Pipeline::gesTrainCams(RawGaussianModel &model, std::vector<Camera> cams)
{
    assert(model.getRenderMethod() == "ges");
    std::cout << "loading ref images...\n";
    // 读取mesh render的结果
    for (auto &cam : cams)
    {
        std::string mesh_rgb_name = config["mesh_rgb_path"].as<std::string>() + "/frame" + cam.getFrameID() + ".color.jpg";
        cv::Mat mesh_rgb_img = imreadRGB(mesh_rgb_name);
        torch::Tensor mesh_rgb_tensor = imageToTensor(mesh_rgb_img);

        std::string mesh_depth_name = config["mesh_depth_path"].as<std::string>() + "/frame" + cam.getFrameID() + ".depth.png";
        cv::Mat mesh_depth_img = imread(mesh_depth_name, cv::IMREAD_UNCHANGED);
        mesh_depth_img.convertTo(mesh_depth_img, CV_32F);

        mesh_depth_img /= 1000; // 固定保存深度单位为mm
        torch::Tensor mesh_depth_tensor = depthToTensor(mesh_depth_img);
        mesh_rgbs.emplace_back(mesh_rgb_tensor);
        mesh_depths.emplace_back(mesh_depth_tensor);
    }
    std::cout << "finish\n";
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
    dist = std::uniform_int_distribution<std::size_t>(0, cams.size() - 1);
    model.setParamsDevice(model.device);
    model.setParamsRequireGrad();
    model.initOptimizers(max_iterations, scene_scale);
    RandomSelector data_loader(cams);

    std::cout << "start training" << std::endl;
    // 开始优化
    for (; curr_iter < max_iterations; curr_iter++)
    {
        Camera cam;
        int cam_idx;
        // 随机选择优化的相机
        if (selected_cam_idx == -1)
        {
            auto sample = data_loader.getNext();
            cam = sample.value;
            cam_idx = sample.originalIndex;
        }
        else
        {
            cam = cams[selected_cam_idx];
            cam_idx = selected_cam_idx;
        }

        auto mesh_depth = mesh_depths[cam_idx].to(model.device);
        auto mesh_rgb = mesh_rgbs[cam_idx].to(model.device);
        auto cam_depth = cam.depth.to(model.device);
        auto cam_rgb = cam.image.to(model.device);
        // 根据iter逐渐提升SH
        model.updateSH(curr_iter);

        // 渲染
        logStartTime();
        auto render_res = model.forward(cam, mesh_depth, mesh_rgb);
        // auto render_res = model.forward(cam, cam_depth, cam_rgb);
        TensorDict loss = model.computeLoss(render_res, cam, weight_configs);
        loss["total"].backward();
        logEndTime();

        model.optimizersStep();
        model.optimizersZeroGrad();
        model.schedulersStep();
        // tensorboard记录结果
        TensorDict log_info = loss;
        auto duration = getLogTime();
        log_info["opt_state/opt_gs_num"] = torch::tensor(model.getGaussianNum()).to(torch::kInt32);
        log_info["opt_state/iter_time"] = torch::tensor(duration).to(torch::kFloat32);
        log_info["opt_state/opacity_mean"] = model.getRealOpacities().mean();
        logResults(render_res, log_info, cam, {"rgb"});

        // 更新progress bar的进度
        bar.set_option(option::PostfixText{"Training iter: " + std::to_string(curr_iter) + "/" + std::to_string(max_iterations) + " Loss: " + std::to_string(loss["total"].item<float>())});
        bar.set_progress(100 * curr_iter / max_iterations);
    }
}

void Pipeline::renderEvalImgs(RawGaussianModel &model, const std::vector<Camera> &cams, std::vector<std::string> names)
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
    int read_count = 0;

    for (Camera cam : cams)
    {
        TensorDict render_res;
        if (model.getRenderMethod() == "ges")
        {
            auto mesh_depth = mesh_depths[read_count].to(model.device);
            auto mesh_rgb = mesh_rgbs[read_count].to(model.device);
            auto cam_depth = cam.depth.to(model.device);
            render_res = model.forward(cam, mesh_depth, mesh_rgb);
            // render_res = model.forward(cam, cam.depth, mesh_rgb);
            // std::cout << cam.id << std::endl;
            // std::cout << model.getExposure()[cam.id] << std::endl;
            // TensorDict loss = model.computeLoss(render_res, cam, weight_configs);
        }
        else if (model.getRenderMethod() == "raw")
            render_res = model.forward(cam);
        std::string frame_file_name = "frame" + cam.getFrameID() + "_iter" + std::to_string(curr_iter);
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
                cv::imwrite((fs::path(eval_path) / "render" / (frame_file_name + ".depth.jpg")).string(),
                            rendered_img);
                if (cam.has_depth)
                {
                    torch::Tensor gt_depth = cam.depth;
                    // cv::Mat gt_depth_img = tensorToJetMat(gt_depth, 0, depth_vis_max, true);
                    // cv::imwrite((fs::path(eval_path) / "gt" / (frame_file_name + ".depth.jpg")).string(),
                    //             rendered_img);
                    // save compare image
                    cv::Mat comp_depth = depthCompImg(rendered_depth, gt_depth, 0, depth_vis_max, 0, depth_error_max);
                    cv::imwrite((fs::path(eval_path) / "comp" / (frame_file_name + ".depth.jpg")).string(),
                                comp_depth);
                }
            }
        }
        read_count++;
        bar.set_option(option::PostfixText{"Eval camera: " + frame_file_name});
        bar.set_progress(100 * read_count / cams.size());
    }
    show_console_cursor(true);
}