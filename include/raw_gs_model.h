#pragma once
#include <chrono>
#include "raw_gs_param.h"
#include "dataset_reader.h"

#include "optim_scheduler.hpp"

class RawGaussianModel
{
public:
    RawGaussianModel() = default;

    ~RawGaussianModel() = default;

    // 初始化高斯参数
    void initGaussianParams(const Points &init_points, int frame_num)
    {
        opt_gs_params.init(init_points.xyz, init_points.rgb, init_points.normal, maxSH, defaultOpacities, maxInitScale, minInitScale, frame_num);
        opt_gs_params.toGPU();
    }

    // 读取配置文件设置参数
    void loadConfig(const YAML::Node &config);

    // 根据当前的iteration更新启用的SH阶数
    void updateSH(int curr_iter = -1)
    {
        if (curr_iter >= 0 && shDegreeInterval > 0)
            degreesToUse = (std::min<int>)(maxSH, curr_iter / shDegreeInterval);
        else
            degreesToUse = maxSH;
    }

    // 使用指定的渲染方法进行前向计算
    TensorDict forward(const Camera &cam, const torch::Tensor &ref_depth = torch::Tensor(), const torch::Tensor &base_color = torch::Tensor())
    {
        if (render_method == "raw")
            return rawForward(cam);
        else if (render_method == "ges")
            return gesForward(cam, ref_depth, base_color);
        else
        {
            std::cout << "UNSUPPORTTED RENDER METHOD!!!\t" << render_method << std::endl;
            TensorDict empty_res;
            return empty_res;
        }
    }

    // 一次标准的前向计算
    TensorDict rawForward(const Camera &cam);

    // 一次标准的前向计算
    TensorDict gesForward(const Camera &cam, const torch::Tensor &ref_depth, const torch::Tensor &base_color);

    // 根据前向结果计算误差
    TensorDict computeLoss(TensorDict &render_res, const Camera &cam, const YAML::Node &weight_configs, const torch::Tensor &mask = torch::Tensor());

    // 进行训练后处理操作，记录梯度等信息用来densification
    void stepPostBackward(TensorDict &render_res, const Camera &cam, float scene_scale, int curr_iter);

    // 更新记录的梯度
    void updateDensifyGrad(TensorDict &render_res, const Camera &cam);

    // 进行densify操作，寻找梯度较大的高斯，分裂太大的高斯、克隆小的高斯
    void densifiyGs(float scene_scale, int curr_iter);

    // 进行prune操作，删掉不透明度太低或者screen size太大的高斯
    void prunePoints(const torch::Tensor &deleteMask);

    // 目前此函数只读取了模型参数没有读取优化器参数，只支持从头训练
    void loadParamsTensor(const std::string &filename);

    /////////////////////  SAVE AND LOAD  /////////////////////
    void saveParamsPly(const std::string &filename)
    {
        getGaussianParms().savePly(filename);
    }

    void saveParamsTensor(const std::string &filename)
    {
        getGaussianParms().saveTensor(filename);
    }

    /////////////////////  SET FUNCTIONS  /////////////////////
    void setParamsDevice(torch::Device device)
    {
        opt_gs_params.toDevice(device);
    }

    void setParamsRequireGrad()
    {
        opt_gs_params.requireGrad(true);
    }

    ////////////////////  Optimizer Functions ////////////////////

    void initOptimizers(int max_iterations = -1, float scene_scale = 1);

    void optimizersZeroGrad();

    // 根据优化器更新参数
    void optimizersStep();

    void addToOptimizer(torch::optim::Adam *optimizer, const torch::Tensor &newParam, const torch::Tensor &idcs, int nSamples);

    void removeFromOptimizer(torch::optim::Adam *optimizer, const torch::Tensor &newParam, const torch::Tensor &deletedMask);

    void replaceToOptimizer(torch::optim::Adam *optimizer, const torch::Tensor &newParam);

    void checkOpimizer(torch::optim::Adam *optimizer)
    {
        torch::Tensor param = optimizer->param_groups()[0].params()[0];
        std::cout << "Requires grad: " << param.requires_grad() << std::endl;
        std::cout << "Has grad: " << param.grad().defined() << std::endl;
#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR > 1
        auto pId = param.unsafeGetTensorImpl();
#else
        auto pId = c10::guts::to_string(param.unsafeGetTensorImpl());
#endif
        if (optimizer->state().find(pId) != optimizer->state().end())
        {
            auto &state = optimizer->state()[pId];
            if (auto *adamState = dynamic_cast<torch::optim::AdamParamState *>(state.get()))
            {
                // 现在可以安全地使用 adamState
                std::cout << "Found Adam state" << std::endl;
            }
            else
            {
                std::cout << "State is not AdamParamState" << std::endl;
            }
        }
        else
        {
            std::cout << "No state found for this parameter" << std::endl;
        }
        bool found = false;
        for (const auto &group : optimizer->param_groups())
        {
            for (const auto &p : group.params())
            {
                if (p.is_same(param))
                {
                    found = true;
                    break;
                }
            }
            if (found)
                break;
        }
        std::cout << "Parameter is in optimizer: " << found << std::endl;

        for (const auto &group : optimizer->param_groups())
        {
            for (const auto &p : group.params())
            {
                auto pImpl = p.unsafeGetTensorImpl();
                if (optimizer->state().find(pImpl) != optimizer->state().end())
                {
                    std::cout << "Found state for a parameter" << std::endl;
                }
            }
        }
    }

    // 更新学习率
    void schedulersStep()
    {
        if (meansOptScheduler != nullptr)
        {
            meansOptScheduler->step();
        }
    }

    /////////////////////  GET FUNCTIONS  /////////////////////
    RawGaussianParams getGaussianParms()
    {
        return opt_gs_params;
    }

    int getMaxSH() const
    {
        return maxSH;
    }

    // get functions
    torch::Tensor getMeans()
    {
        return opt_gs_params.means;
    }

    torch::Tensor getScales()
    {
        return opt_gs_params.scales;
    }

    torch::Tensor getQuats()
    {
        return opt_gs_params.quats;
    }

    torch::Tensor getFeaturesDc()
    {
        return opt_gs_params.featuresDc;
    }

    torch::Tensor getFeaturesRest()
    {
        return opt_gs_params.featuresRest;
    }

    torch::Tensor getOpacities()
    {
        return opt_gs_params.opacities;
    }

    torch::Tensor getExposure()
    {
        return opt_gs_params.exposure;
    }

    torch::Tensor getRealMeans()
    {
        return opt_gs_params.getRealMeans();
    }

    torch::Tensor getRealScales()
    {
        return opt_gs_params.getRealScales();
    }

    torch::Tensor getRealOpacities()
    {
        return opt_gs_params.getRealOpacities();
    }

    int getGaussianNum()
    {
        return opt_gs_params.getGaussianNum();
    }

    std::string getRenderMethod()
    {
        return render_method;
    }

    RawGaussianParams opt_gs_params;
    // 数据在哪个设备上(gpu/cpu)，目前默认只支持gpu
    torch::Device device = torch::kCUDA;

    // 高斯参数优化器
    torch::optim::Adam *meansOpt = nullptr;
    torch::optim::Adam *scalesOpt = nullptr;
    torch::optim::Adam *quatsOpt = nullptr;
    torch::optim::Adam *featuresDcOpt = nullptr;
    torch::optim::Adam *featuresRestOpt = nullptr;
    torch::optim::Adam *opacitiesOpt = nullptr;
    torch::optim::Adam *exposureOpt = nullptr;
    OptimScheduler *meansOptScheduler = nullptr;

private:
    friend class SLAMGaussianModel;
    YAML::Node config;
    // 高斯的SH阶数, 最大阶数，每隔多少iter增加SH
    int degreesToUse, maxSH, shDegreeInterval;
    float maxInitScale, minInitScale;
    // 默认初始不透明度(0-1)，-1表示随机初始化
    float defaultOpacities;
    // 各个属性的学习率
    float meansLr, scalesLr, quatsLr, featuresDcLr, featuresRestLr, opacitiesLr, exposureLr;
    // 是否使用曝光优化
    bool use_exposure;

    // Densify有关的参数和变量
    int lastWidth, lastHeight;
    torch::Tensor grad_2d, visible_count, max2DSize;
    int refine_scale2d_stop_iter = 0;
    int pause_refine_after_reset = 0;

    // 渲染的默认参数，参考gsplat设置
    std::string render_method = "raw";
    // FullyFusedProjection
    float eps2d = 0.3;
    float near_plane = 0.01;
    float far_plane = 1e10;
    float radius_clip = 0.0;
    bool calc_compensations = false;
    std::string camera_model = "pinhole";
    int max_gs_radii = -1;
    // isectTiles
    int tile_size = 16;
    // RasterizeToPixels
    bool abs_grad = false;
    float delta_depth = 0;
    at::optional<torch::Tensor> backgrounds;
    // 背景颜色
    torch::Tensor backgroundColor = torch::tensor({0.0f, 0.0f, 0.0f}, torch::kCUDA);
};