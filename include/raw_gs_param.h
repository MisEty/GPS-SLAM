#pragma once

#include <torch/torch.h>
#include "gsplat_wapper.hpp"
#include "file_utils.h"

class RawGaussianParams
{
    friend class RawGaussianModel;

protected:
    torch::Tensor means;
    torch::Tensor scales;
    torch::Tensor quats;
    torch::Tensor featuresDc;
    torch::Tensor featuresRest;
    torch::Tensor opacities;
    torch::Tensor exposure;

public:
    RawGaussianParams() = default;

    ~RawGaussianParams() = default;

    void init(const torch::Tensor &xyz,
              const torch::Tensor &rgb,
              const torch::Tensor &normals,
              int max_sh_degree,
              float init_opacs,
              float max_scale = -1,
              float min_scale = -1,
              int exposure_num = 1);

    bool isDefined() const { return means.defined(); }

    void requireGrad(bool require_grad);

    void toGPU();

    void toCPU();

    void toDevice(torch::Device device);

    void add(const RawGaussianParams &other);

    void remove(const torch::Tensor &mask);

    void savePly(const std::string &filename);

    void saveTensor(const std::string &filename);

    void loadTensor(const std::string &filename);

    // 直接访问tensor的原始值
    int getGaussianNum()
    {
        if (!isDefined())
            return 0;
        else
            return means.size(0);
    }

    torch::Tensor getMeans() { return means; }

    torch::Tensor getQuats() { return quats; }

    torch::Tensor getScales() { return scales; }

    torch::Tensor getFeaturesDc() { return featuresDc; }

    torch::Tensor getFeaturesRest() { return featuresRest; }

    torch::Tensor getOpacities() { return opacities; }

    torch::Tensor getExposure() { return exposure; }

    // 访问tensor具有实际意义的值
    torch::Tensor getRealMeans() { return means; }

    torch::Tensor getRealScales() { return torch::exp(scales); }

    torch::Tensor getRealOpacities() { return torch::sigmoid(opacities); }

    // void printInfos();
};