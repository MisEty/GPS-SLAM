#include "pipeline.h"
#include "slam_gs_model.h"
#include "InfiniTAM_tools.h"
#include <deque>

class SLAMPipeline : public Pipeline
{
public:
    void loadConfig(const YAML::Node &config, const std::string &workspace_dir, bool is_train);

    void SLAMTrainCams(SLAMGaussianModel &model, std::vector<Camera> &cams);

    void localOptimize(SLAMGaussianModel &model);

    void removeRedundantGs(SLAMGaussianModel &model);

    void updateFrameList();

    void localFrameRaycast();

    void initNewGaussians(SLAMGaussianModel &model, TensorDict &raycast_maps);

    void setTsdfEngine(CLIEngine *tsdf_engine)
    {
        this->tsdf_engine = tsdf_engine;
        this->main_engine = tsdf_engine->getMainEngine();
        raycast_color_raw_temp = new ITMUChar4Image(tsdf_engine->GetDepthSize(), true, true);
        raycast_vertex_raw_temp = new ITMFloat4Image(tsdf_engine->GetDepthSize(), true, true);
        voxel_size = dynamic_cast<ITMBasicEngine<ITMVoxel, ITMVoxelIndex> *>(main_engine)->getVoxelSize();
    }

    void saveMesh()
    {
        if (config["TSDF"]["saved_mesh"].as<std::string>() == "")
            return;
        tsdf_engine->getMainEngine()->SaveSceneToMesh((workspace_dir + "/" + config["TSDF"]["saved_mesh"].as<std::string>()).c_str());
    }

    void saveEngine()
    {
        if (config["TSDF"]["saved_engine"].as<std::string>() == "")
            return;
        dynamic_cast<ITMBasicEngine<ITMVoxel, ITMVoxelIndex> *>(tsdf_engine->getMainEngine())->SaveToFile(workspace_dir + "/" + config["TSDF"]["saved_engine"].as<std::string>());
    }

    void loadEngine()
    {
        dynamic_cast<ITMBasicEngine<ITMVoxel, ITMVoxelIndex> *>(tsdf_engine->getMainEngine())->LoadFromFile(workspace_dir + "/" + config["TSDF"]["saved_engine"].as<std::string>());
    }

    // 输入任意dataset内的相机，使用main engine里预先存好的cam pose进行raycast，返回深度图和颜色图
    TensorDict runRaycastByCam(const Camera &cam, bool use_cam_depth = true);

    void renderEvalImgs(SLAMGaussianModel &model, const std::vector<Camera> &cams, std::vector<std::string> names);

    void keyFrameRaycast(SLAMGaussianModel &model);

    void checkKeyFrameError(SLAMGaussianModel &model);

    // private:
    torch::Device device = torch::kCUDA;
    std::string work_mode = "train";
    int device_id = 0;

    CLIEngine *tsdf_engine;
    ITMMainEngine *main_engine;
    float voxel_size;
    ITMUChar4Image *raycast_color_raw_temp;
    ITMFloat4Image *raycast_vertex_raw_temp;
    cv::Mat raycast_color_mat, raycast_vertex_mat;

    int curr_frame_id = -1;
    int localframe_cam_window_length, localframe_cam_window_interval;
    int local_opt_iters, local_opt_interval;
    int keyframe_select_max = 2;
    Camera curr_cam;

    std::deque<Camera> localframe_cam_window;
    std::deque<TensorDict> localframe_raycast_window;
    std::map<std::string, std::vector<float>> keyframe_loss_dict;
    std::vector<Camera> keyframe_cam_list;
    std::map<std::string, Camera> keyframe_cam_dict;
    std::vector<Camera> opt_cam_list;
    std::vector<TensorDict> opt_raycast_list;

    float keyframe_theta_thres;
    float keyframe_trans_thres;
    bool is_keyframe = false;
    bool log_slam_state;

    float new_gs_sample_ratio;
    float empty_alpha_thres;
    float color_error_thres;
};