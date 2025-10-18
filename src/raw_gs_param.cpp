#include "raw_gs_param.h"
#include "kdtree_tensor.h"
#include "tensor_math.h"
#include "gsplat_wapper.hpp"
#include <fstream>

using namespace torch::indexing;

// #define LOG_PARAM_TIME

void RawGaussianParams::init(const torch::Tensor &xyz,
                             const torch::Tensor &rgb,
                             const torch::Tensor &normals,
                             int max_sh_degree,
                             float init_opacs,
                             float max_scale,
                             float min_scale,
                             int exposure_num)
{
#ifdef LOG_PARAM_TIME
    struct timespec scale_compute_start, scale_compute_end, quat_compute_end, sh_compute_end, else_compute_end;
    clock_gettime(CLOCK_MONOTONIC, &scale_compute_start);
#endif
    long long points_num = xyz.size(0);
    // set means
    this->means = xyz;
    // 按照KNN距离计算高斯尺度
    torch::Tensor raw_scales = torch::sqrt(distCUDA2(xyz)).clamp(min_scale, max_scale).unsqueeze(1).repeat({1, 3});
#ifdef LOG_PARAM_TIME
    clock_gettime(CLOCK_MONOTONIC, &scale_compute_end);
#endif
    // 随机初始化旋转向量
    // auto raw_quats = randomQuatTensor(points_num);
    auto raw_quats = torch::ones({points_num, 4}).to(torch::kCUDA);
    if (normals.defined())
    {
        // 有法向量的情况下，将z轴设为最短轴(0.1)并且使其指向法向量
        raw_scales.index_put_({torch::indexing::Slice(), 2}, raw_scales.index({torch::indexing::Slice(), 2}) * 0.1);
        // compute rotation and scales according to the normals
        torch::Tensor z_axis = torch::zeros_like(raw_scales).to(torch::kCUDA);
        z_axis.index_put_({torch::indexing::Slice(), 2}, 1);
        raw_quats = computeQuat(z_axis, normals);
    }
    this->scales = raw_scales.log();
    this->quats = raw_quats;
#ifdef LOG_PARAM_TIME
    clock_gettime(CLOCK_MONOTONIC, &quat_compute_end);
#endif
    // 从RGB颜色计算SH
    int sh_dims = numShBases(max_sh_degree);
    torch::Tensor shs = torch::zeros({points_num, sh_dims, 3},
                                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    shs.index({Slice(), 0, Slice(None, 3)}) = rgb2sh(rgb);
    // shs.index({Slice(), Slice(1, None), Slice(3, None)}) = 0.0f;
#ifdef LOG_PARAM_TIME
    clock_gettime(CLOCK_MONOTONIC, &sh_compute_end);
#endif
    this->featuresDc = shs.index({Slice(), 0, Slice()});
    this->featuresRest = shs.index({Slice(), Slice(1, None), Slice()});
    this->opacities = torch::logit(init_opacs * torch::ones({points_num, 1}));
    if (exposure_num > 0)
    {
        torch::Tensor init_exposure = torch::eye(3, 4, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        this->exposure = init_exposure.unsqueeze(0).repeat({exposure_num, 1, 1});
    }
#ifdef LOG_PARAM_TIME
    clock_gettime(CLOCK_MONOTONIC, &else_compute_end);
    printf("[PARAM TIME] scale_compute: %f, quat_compute: %f, sh_compute: %f, else_compute: %f\n",
           calculateTimeInterval(scale_compute_start, scale_compute_end),
           calculateTimeInterval(scale_compute_end, quat_compute_end),
           calculateTimeInterval(quat_compute_end, sh_compute_end),
           calculateTimeInterval(sh_compute_end, else_compute_end));
#endif
}

void RawGaussianParams::requireGrad(bool require_grad)
{
    means.set_requires_grad(require_grad);
    scales.set_requires_grad(require_grad);
    quats.set_requires_grad(require_grad);
    featuresDc.set_requires_grad(require_grad);
    featuresRest.set_requires_grad(require_grad);
    opacities.set_requires_grad(require_grad);
    exposure.set_requires_grad(require_grad);
}

void RawGaussianParams::toGPU()
{
    torch::Device device = torch::kCUDA;
    means = means.to(device);
    scales = scales.to(device);
    quats = quats.to(device);
    featuresDc = featuresDc.to(device);
    featuresRest = featuresRest.to(device);
    opacities = opacities.to(device);
    exposure = exposure.to(device);
}

void RawGaussianParams::toCPU()
{
    torch::Device device = torch::kCPU;
    means = means.to(device);
    scales = scales.to(device);
    quats = quats.to(device);
    featuresDc = featuresDc.to(device);
    featuresRest = featuresRest.to(device);
    opacities = opacities.to(device);
    exposure = exposure.to(device);
}

// 这个函数会保证改变device之后还是叶子节点，可以用来优化
void RawGaussianParams::toDevice(torch::Device device)
{
    means = means.detach().clone().to(device);
    scales = scales.detach().clone().to(device);
    quats = quats.detach().clone().to(device);
    featuresDc = featuresDc.detach().clone().to(device);
    featuresRest = featuresRest.detach().clone().to(device);
    opacities = opacities.detach().clone().to(device);
    exposure = exposure.detach().clone().to(device);
}

void RawGaussianParams::add(const RawGaussianParams &other)
{
    if (!isDefined())
    {
        means = other.means;
        scales = other.scales;
        quats = other.quats;
        featuresDc = other.featuresDc;
        featuresRest = other.featuresRest;
        opacities = other.opacities;
        exposure = other.exposure;
    }
    else
    {
        means = torch::cat({means, other.means}, 0);
        scales = torch::cat({scales, other.scales}, 0);
        quats = torch::cat({quats, other.quats}, 0);
        featuresDc = torch::cat({featuresDc, other.featuresDc}, 0);
        featuresRest = torch::cat({featuresRest, other.featuresRest}, 0);
        opacities = torch::cat({opacities, other.opacities}, 0);
        exposure = torch::cat({exposure, other.exposure}, 0);
    }
}

// 根据mask删除指定的高斯
void RawGaussianParams::remove(const torch::Tensor &mask)
{
    means = means.index({~mask}).detach().requires_grad_();
    scales = scales.index({~mask}).detach().requires_grad_();
    quats = quats.index({~mask}).detach().requires_grad_();
    featuresDc = featuresDc.index({~mask}).detach().requires_grad_();
    featuresRest = featuresRest.index({~mask}).detach().requires_grad_();
    opacities = opacities.index({~mask}).detach().requires_grad_();
    // exposure = exposure.index({~mask}).detach().requires_grad_();
}

void RawGaussianParams::savePly(const std::string &filename)
{
    std::ofstream o(filename, std::ios_base::out);
    int numPoints = getGaussianNum();

    o << "ply" << std::endl;
    o << "format binary_little_endian 1.0" << std::endl;
    o << "element vertex " << numPoints << std::endl;
    o << "property float x" << std::endl;
    o << "property float y" << std::endl;
    o << "property float z" << std::endl;
    o << "property float nx" << std::endl;
    o << "property float ny" << std::endl;
    o << "property float nz" << std::endl;

    for (int i = 0; i < featuresDc.size(1); i++)
    {
        o << "property float f_dc_" << i << std::endl;
    }
    torch::Tensor featuresRestCpu = featuresRest.cpu().transpose(1, 2).reshape({numPoints, -1});
    for (int i = 0; i < featuresRestCpu.size(1); i++)
    {
        o << "property float f_rest_" << i << std::endl;
    }

    o << "property float opacity" << std::endl;

    o << "property float scale_0" << std::endl;
    o << "property float scale_1" << std::endl;
    o << "property float scale_2" << std::endl;

    o << "property float rot_0" << std::endl;
    o << "property float rot_1" << std::endl;
    o << "property float rot_2" << std::endl;
    o << "property float rot_3" << std::endl;

    o << "end_header" << std::endl;

    float zeros[] = {0.0f, 0.0f, 0.0f};

    torch::Tensor meansCpu = means.cpu();
    torch::Tensor featuresDcCpu = featuresDc.cpu();
    torch::Tensor opacitiesCpu = opacities.cpu();
    torch::Tensor scalesCpu = scales.cpu();
    torch::Tensor quatsCpu = quats.cpu();

    for (size_t i = 0; i < numPoints; i++)
    {
        o.write(reinterpret_cast<const char *>(meansCpu[i].data_ptr()), sizeof(float) * 3);
        o.write(reinterpret_cast<const char *>(zeros), sizeof(float) * 3);
        o.write(reinterpret_cast<const char *>(featuresDcCpu[i].data_ptr()), sizeof(float) * featuresDcCpu.size(1));
        o.write(reinterpret_cast<const char *>(featuresRestCpu[i].data_ptr()), sizeof(float) * featuresRestCpu.size(1));
        o.write(reinterpret_cast<const char *>(opacitiesCpu[i].data_ptr()), sizeof(float) * 1);
        o.write(reinterpret_cast<const char *>(scalesCpu[i].data_ptr()), sizeof(float) * 3);
        o.write(reinterpret_cast<const char *>(quatsCpu[i].data_ptr()), sizeof(float) * 4);
    }

    o.close();
    std::cout << "Wrote ply model to: " << filename << std::endl;
}

void RawGaussianParams::saveTensor(const std::string &filename)
{
    std::unordered_map<std::string, torch::Tensor> tensor_map;
    tensor_map["means"] = means;
    tensor_map["scales"] = scales;
    tensor_map["quats"] = quats;
    tensor_map["featuresDc"] = featuresDc;
    tensor_map["featuresRest"] = featuresRest;
    tensor_map["opacities"] = opacities;
    tensor_map["exposure"] = exposure;

    torch::serialize::OutputArchive archive;
    for (const auto &pair : tensor_map)
    {
        archive.write(pair.first, pair.second);
    }
    archive.save_to(filename);
    std::cout << "Wrote tensor model to: " << filename << std::endl;
}

void RawGaussianParams::loadTensor(const std::string &filename)
{
    torch::serialize::InputArchive archive;
    archive.load_from(filename);

    archive.read("means", means);
    archive.read("scales", scales);
    archive.read("quats", quats);
    archive.read("featuresDc", featuresDc);
    archive.read("featuresRest", featuresRest);
    archive.read("opacities", opacities);
    archive.read("exposure", exposure);

    std::cout << "Loaded tensors from: " << filename << std::endl;
}
