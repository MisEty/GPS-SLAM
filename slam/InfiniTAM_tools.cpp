#include "InfiniTAM_tools.h"

CLIEngine *createTsdfEngine(const DatasetReader &data_reader, const YAML::Node &config)
{
    // 1. 建立InifiTAM Calib参数
    ITMRGBDCalib rgbd_calib;
    rgbd_calib.intrinsics_rgb.SetFrom(data_reader.width, data_reader.height, data_reader.fx, data_reader.fy, data_reader.cx, data_reader.cy);
    rgbd_calib.intrinsics_d = rgbd_calib.intrinsics_rgb;
    // 默认深度转换比例为1000, 且无偏移
    rgbd_calib.disparityCalib.SetStandard();

    // 2. 创建IfiniTAM images;
    int image_num = data_reader.train_vec.size();
    std::vector<ITMUChar4Image *> rgb_images(image_num);
    std::vector<ITMShortImage *> depth_images(image_num);
    std::vector<ORUtils::Matrix4<float> *> gt_c2w_poses(image_num);

    show_console_cursor(false);
    ProgressBar bar{
        option::BarWidth{50},
        option::Start{"["},
        option::Fill{"="},
        option::Lead{">"},
        option::Remainder{" "},
        option::End{"]"},
        option::PostfixText{"Convert InfiniTAM format"},
        option::ForegroundColor{Color::green},
        option::ShowPercentage{true},
        option::ShowElapsedTime{true},
        option::ShowRemainingTime{true},
        option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}};

    for (int i = 0; i < image_num; i++)
    {
        cv::Mat rgb_image_cv = tensorToImage(data_reader.train_vec[i].image);
        rgb_images[i] = imageToITMUChar4Image(rgb_image_cv);

        cv::Mat depth_image_cv = tensorToDepth(data_reader.train_vec[i].depth);
        depth_images[i] = depthToITMUShortImage(depth_image_cv);

        gt_c2w_poses[i] = tensorToInfiMatrix4(data_reader.train_vec[i].c2w);

        bar.set_option(option::PostfixText{"Convert camera: " + std::to_string(i) + "/" + std::to_string(image_num)});
        bar.set_progress(100 * i / image_num);
    }
    show_console_cursor(true);
    // 3. 创建main Engine
    ITMLibSettings *internalSettings = new ITMLibSettings();
    internalSettings->sceneParams.voxelSize = config["voxel_size"].as<float>();
    internalSettings->sceneParams.mu = config["trunc_dist"].as<float>();
    internalSettings->sceneParams.viewFrustum_min = config["viewFrustum_min"].as<float>();
    internalSettings->sceneParams.viewFrustum_max = config["viewFrustum_max"].as<float>();
    ITMMainEngine *mainEngine = new ITMBasicEngine<ITMVoxel, ITMVoxelIndex>(
        internalSettings,
        rgbd_calib,
        rgb_images[0]->noDims,
        depth_images[0]->noDims);

    if (config["use_gt_pose"].as<bool>())
    {
        dynamic_cast<ITMBasicEngine<ITMVoxel, ITMVoxelIndex> *>(mainEngine)->turnOffTracking();
        dynamic_cast<ITMBasicEngine<ITMVoxel, ITMVoxelIndex> *>(mainEngine)->gtC2wPoses = gt_c2w_poses;
    }
    // 4. 创建CLI Engine
    CLIEngine *tsdf_engine = CLIEngine::Instance();
    tsdf_engine->Initialise(rgb_images, depth_images, mainEngine);
    return tsdf_engine;
}