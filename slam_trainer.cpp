#include "dataset_reader.h"
#include "slam_pipeline.h"
#include "InfiniTAM_tools.h"

int main(int argc, char *argv[])
{
    std::cout << "ours trainer demo!" << std::endl;
    const char *config_filename = argv[1];
    YAML::Node config = YAML::LoadFile(config_filename);
    std::string workspace_dir = config["workspace_dir"].as<std::string>();
    std::string work_mode = config["work_mode"].as<std::string>();

    // setup cout precision
    std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);

    // setup cuda device
    const std::string devId = config["dev_id"].as<std::string>();
    setenv("CUDA_VISIBLE_DEVICES", devId.c_str(), 1);

    std::cout << "======= load camera data ======" << std::endl;
    DatasetReader data_reader(config["READER"]);
    data_reader.read();
    data_reader.updateSceneGeo();

    std::cout << "======= Convert InfiniTAM data ======" << std::endl;
    CLIEngine *tsdf_engine = createTsdfEngine(data_reader, config["PIPE"]["TSDF"]);
    SLAMPipeline pipe;
    pipe.setTsdfEngine(tsdf_engine);
    pipe.work_mode = work_mode;
    pipe.device_id = config["dev_id"].as<int>();

    if (work_mode == "train" || work_mode == "recon")
    {
        std::cout << "======= setup Gaussian model ======" << std::endl;
        SLAMGaussianModel model;
        model.loadConfig(config["MODEL"]);

        std::cout << "======= setup training pipe ======" << std::endl;
        pipe.scene_scale = data_reader.scene_scale;
        createWorkSpace(config_filename);
        pipe.loadConfig(config["PIPE"], workspace_dir, true);

        std::cout << "======= start training ======" << std::endl;
        pipe.SLAMTrainCams(model, data_reader.train_vec);
        std::cout << "train finish" << std::endl;
        std::cout << "keyframe num: " << pipe.keyframe_cam_list.size() << std::endl;
        if (pipe.save_after_train)
        {
            std::cout << "======= save model, mesh and tsdf ======" << std::endl;
            pipe.save(model, data_reader.getAllCams());
            pipe.saveEngine();
            pipe.saveMesh();
            data_reader.savePose(pipe.eval_path + "/pose");
        }

        if (pipe.eval_after_train)
        {
            std::cout << "======= render eval images ======" << std::endl;
            pipe.renderEvalImgs(model, data_reader.train_vec, {"rgb"});
        }
    }

    else if (work_mode == "eval")
    {
        std::cout << "======= setup Gaussian model ======" << std::endl;
        SLAMGaussianModel model;
        model.loadConfig(config["MODEL"]);
        model.loadParamsTensor(workspace_dir + "/gs_model/model.pt");
        std::cout << "======= setup eval pipe ======" << std::endl;
        pipe.scene_scale = data_reader.scene_scale;
        pipe.loadConfig(config["PIPE"], workspace_dir, false);
        pipe.loadEngine();
        std::cout << "======= start eval ======" << std::endl;
        pipe.renderEvalImgs(model, data_reader.train_vec, {"rgb"});
    }
}
