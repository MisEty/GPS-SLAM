#include <c10/cuda/CUDACachingAllocator.h>

#include "raw_gs_model.h"
#include "tensor_math.h"

using namespace torch::indexing;
using namespace torch::autograd;

// #define LOGFORWARDTIME

void RawGaussianModel::loadConfig(const YAML::Node &config)
{
    std::cout << "loading Gaussian model parameters...";
    this->config = config;
    // Gaussian setup
    render_method = config["render_method"].as<std::string>();
    maxSH = config["sh_degree"].as<int>();
    maxInitScale = config["max_init_scale"].as<float>();
    minInitScale = config["min_init_scale"].as<float>();

    // 默认使用最大SH进行渲染和训练
    degreesToUse = maxSH;
    shDegreeInterval = config["sh_degree_interval"].as<int>();
    defaultOpacities = config["default_opacities"].as<float>();
    // learning rate
    meansLr = config["means_lr"].as<float>();
    scalesLr = config["scales_lr"].as<float>();
    quatsLr = config["quats_lr"].as<float>();
    featuresDcLr = config["featuresDc_lr"].as<float>();
    featuresRestLr = config["featuresRest_lr"].as<float>();
    exposureLr = config["exposure_lr"].as<float>();
    use_exposure = config["use_exposure"].as<bool>();
    if (featuresRestLr < 0)
        featuresRestLr = featuresDcLr / 20;
    opacitiesLr = config["opacities_lr"].as<float>();
    // rendering set up
    max_gs_radii = config["max_gs_radii"].as<int>();
    delta_depth = config["delta_depth"].as<float>();
    std::cout << "finish" << std::endl;
}

// TODO: 目前只考虑一个相机的情况，相机维度使用unsqueeze()添加。如果有多相机需要修改
TensorDict RawGaussianModel::rawForward(const Camera &cam)
{
#ifdef LOGFORWARDTIME
    struct timespec FullyFusedProjection_start, FullyFusedProjection_end,
        ComputeViewDir_start, ComputeViewDir_end,
        SphericalHarmonics_start, SphericalHarmonics_end,
        isectTiles_start, isectTiles_end,
        isectOffsetEncode_start, isectOffsetEncode_end,
        RasterizeToPixels_start, RasterizeToPixels_end;
#endif
    lastHeight = cam.height;
    lastWidth = cam.width;
    auto c2w = cam.c2w.to(device);
    auto cam_T = c2w.index({Slice(None, 3), Slice(3, 4)});
    torch::Tensor Ks = cam.K.to(device);

    at::optional<torch::Tensor> empty_none;
    torch::Tensor viewMat = poseInv(c2w);
    torch::Tensor world_means = getRealMeans().contiguous();
    auto world_covars = empty_none;
    torch::Tensor world_scales = getRealScales().contiguous();
#ifdef LOGFORWARDTIME
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &FullyFusedProjection_start);
#endif
    // STEP1: 高斯投影
    // 注意，相机参数的输入必须是[C, MATRIX_SHAPE]类型的，需要有一个维度表示多相机
    auto proj_res = FullyFusedProjection::apply(world_means,
                                                world_covars,
                                                getQuats(),
                                                world_scales,
                                                viewMat.unsqueeze(0),
                                                Ks.unsqueeze(0),
                                                cam.width,
                                                cam.height,
                                                eps2d,
                                                near_plane,
                                                far_plane,
                                                radius_clip,
                                                calc_compensations,
                                                camera_model);
    torch::Tensor radiis = proj_res[0];
    torch::Tensor means2d = proj_res[1];
    torch::Tensor depths = proj_res[2];
    torch::Tensor conics = proj_res[3];
#ifdef LOGFORWARDTIME
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &FullyFusedProjection_end);
#endif
    // STEP2: 将高斯颜色SH转换成RGB
    torch::Tensor shs = torch::cat({getFeaturesDc().index({Slice(), None, Slice()}), getFeaturesRest()}, 1);

    // TODO: 不知道为什么这个减法运算特别慢，需要检查一下原因，或者直接用cuda实现
    torch::Tensor viewDirs = world_means - cam_T.transpose(0, 1);
    // torch::Tensor viewDirs = world_means;

    auto visible_mask = radiis > 0;
#ifdef LOGFORWARDTIME
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &SphericalHarmonics_start);
#endif
    torch::Tensor colors = SphericalHarmonicsNew::apply(degreesToUse, viewDirs.unsqueeze(0), shs.unsqueeze(0), visible_mask);
    // 这里gsplat会有一些很大的异常值，可能是mask处理后的默认值不同
    colors = torch::clamp_min(colors + 0.5f, 0.0f);
#ifdef LOGFORWARDTIME
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &SphericalHarmonics_end);
#endif
// cudaDeviceSynchronize();
// clock_gettime(CLOCK_MONOTONIC, &SphericalHarmonics_end);
#ifdef LOGFORWARDTIME
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &isectTiles_start);
#endif
    // STEP3: 计算相交的tiles
    int tile_width = std::ceil(float(cam.width) / float(tile_size));
    int tile_height = std::ceil(float(cam.height) / float(tile_size));
    auto isec_res = isectTiles(means2d, radiis, depths, tile_size, tile_width, tile_height);

    torch::Tensor tiles_per_gauss = isec_res[0];
    torch::Tensor isect_ids = isec_res[1];
    torch::Tensor flatten_ids = isec_res[2];
    torch::Tensor isect_offsets = isectOffsetEncode(isect_ids, 1, tile_width, tile_height);
#ifdef LOGFORWARDTIME
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &isectTiles_end);
#endif
    // STEP4: 进行per-pixel render
    auto tile_masks = empty_none;
    colors = torch::cat({colors, depths.unsqueeze(-1)}, 2); // 把深度也进行混合
#ifdef LOGFORWARDTIME
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &RasterizeToPixels_start);
#endif
    auto rasterize_res = RasterizeToPixels::apply(means2d,
                                                  conics,
                                                  colors,
                                                  getRealOpacities(),
                                                  backgrounds,
                                                  tile_masks,
                                                  cam.width,
                                                  cam.height,
                                                  tile_size,
                                                  isect_offsets,
                                                  flatten_ids,
                                                  abs_grad);

    torch::Tensor render_colors = rasterize_res[0];
    torch::Tensor render_alphas = rasterize_res[1];
    // STEP5: 计算加权平均深度
    int64_t last_dim = render_colors.size(-1);
    torch::Tensor rgb = render_colors.slice(-1, 0, last_dim - 1);
    torch::Tensor raw_depth = render_colors.slice(-1, last_dim - 1);
    torch::Tensor expected_depth = raw_depth / render_alphas.clamp(1e-10);
#ifdef LOGFORWARDTIME
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &RasterizeToPixels_end);
#endif
    // rgb = torch::clamp_max(rgb, 1.0f);
    TensorDict res;
    // all render res with shape [N, H, W, C], and we assumue one camera!
    res["rgb"] = rgb[0];
    res["depth"] = expected_depth[0];
    res["alpha"] = render_alphas[0];
    res["radiis"] = radiis[0];
    res["means2d"] = means2d; // 这里必须返回原tensor，切片操作会导致无法记录梯度
#ifdef LOGFORWARDTIME
    auto proj_time = calculateTimeInterval(FullyFusedProjection_start, FullyFusedProjection_end);
    auto sh_time = calculateTimeInterval(SphericalHarmonics_start, SphericalHarmonics_end);
    auto tile_insect_time = calculateTimeInterval(isectTiles_start, isectTiles_end);
    auto rasterize_time = calculateTimeInterval(RasterizeToPixels_start, RasterizeToPixels_end);
    auto forward_time = calculateTimeInterval(FullyFusedProjection_start, RasterizeToPixels_end);
    printf("proj_time: %f, sh_time: %f, tile_insect_time: %f, rasterize_time: %f, forward_time: %f\n",
           proj_time, sh_time, tile_insect_time, rasterize_time, forward_time);
    res["opt_state/proj_time"] = torch::tensor(proj_time);
    res["opt_state/sh_time"] = torch::tensor(sh_time);
    res["opt_state/tile_insect_time"] = torch::tensor(tile_insect_time);

    res["opt_state/rasterize_time"] = torch::tensor(rasterize_time);
    res["opt_state/forward_time"] = torch::tensor(forward_time);
#endif
    return res;
}

// TODO: 目前只考虑一个相机的情况，相机维度使用unsqueeze()添加。如果有多相机需要修改
TensorDict RawGaussianModel::gesForward(const Camera &cam, const torch::Tensor &ref_depth, const torch::Tensor &base_color)
{
#ifdef LOGFORWARDTIME
    struct timespec FullyFusedProjection_start, FullyFusedProjection_end,
        ComputeViewDir_start, ComputeViewDir_end,
        SphericalHarmonics_start, SphericalHarmonics_end,
        isectTiles_start, isectTiles_end,
        isectOffsetEncode_start, isectOffsetEncode_end,
        RasterizeToPixels_start, RasterizeToPixels_end;
#endif
    lastHeight = cam.height;
    lastWidth = cam.width;
    auto c2w = cam.c2w_slam.to(device);
    auto cam_T = c2w.index({Slice(None, 3), Slice(3, 4)});
    torch::Tensor Ks = cam.K.to(device);
    int tile_width = std::ceil(float(cam.width) / float(tile_size));
    int tile_height = std::ceil(float(cam.height) / float(tile_size));
    float min_depth = 0.01;
    float infini_depth = 1000;
    auto ref_depth_clamped = torch::where(ref_depth < 0.01, torch::full_like(ref_depth, infini_depth), ref_depth);


    at::optional<torch::Tensor> empty_none;
    torch::Tensor viewMat = poseInv(c2w);
    torch::Tensor world_means = getRealMeans().contiguous();
    auto world_covars = empty_none;
    torch::Tensor world_scales = getRealScales().contiguous();
#ifdef LOGFORWARDTIME
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &FullyFusedProjection_start);
#endif
    // STEP1: 高斯投影
    // 注意，相机参数的输入必须是[C, MATRIX_SHAPE]类型的，需要有一个维度表示多相机
    auto proj_res = FullyFusedProjection::apply(world_means,
                                                world_covars,
                                                getQuats(),
                                                world_scales,
                                                viewMat.unsqueeze(0),
                                                Ks.unsqueeze(0),
                                                cam.width,
                                                cam.height,
                                                eps2d,
                                                near_plane,
                                                far_plane,
                                                radius_clip,
                                                calc_compensations,
                                                camera_model);

    torch::Tensor radiis = proj_res[0];
    torch::Tensor means2d = proj_res[1];
    torch::Tensor depths = proj_res[2];
    torch::Tensor conics = proj_res[3];

    if (max_gs_radii > 0)
        radiis = torch::clamp_max(radiis, max_gs_radii);
#ifdef LOGFORWARDTIME
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &FullyFusedProjection_end);
#endif

#ifdef LOGFORWARDTIME
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &SphericalHarmonics_start);
#endif
    // STEP2: 将高斯颜色SH转换成RGB
    torch::Tensor shs = torch::cat({getFeaturesDc().index({Slice(), None, Slice()}), getFeaturesRest()}, 1);
    torch::Tensor viewDirs = world_means - cam_T.transpose(0, 1);
    auto visible_mask = radiis > 0;
    torch::Tensor colors = SphericalHarmonicsNew::apply(degreesToUse, viewDirs.unsqueeze(0), shs.unsqueeze(0), visible_mask);
    colors = torch::clamp_min(colors + 0.5f, 0.0f);
#ifdef LOGFORWARDTIME
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &SphericalHarmonics_end);
#endif
// 这里gsplat会有一些很大的异常值，可能是mask处理后的默认值不同

// STEP3: 计算相交的tiles
#ifdef LOGFORWARDTIME
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &isectTiles_start);
#endif
    auto isec_res = isectTilesNoDepth(means2d, radiis, depths, tile_size, tile_width, tile_height, true);

    torch::Tensor tiles_per_gauss = isec_res[0];
    torch::Tensor isect_ids = isec_res[1];
    torch::Tensor flatten_ids = isec_res[2];
    torch::Tensor group_gs_ids = isec_res[3];
    torch::Tensor group_starts = isec_res[4];

    torch::Tensor isect_offsets = isectOffsetEncodeNoDepth(isect_ids, 1, tile_width, tile_height);

#ifdef LOGFORWARDTIME
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &isectTiles_end);
#endif

    // STEP4: 进行per-pixel render
    auto tile_masks = empty_none;
    colors = torch::cat({colors, depths.unsqueeze(-1)}, 2); // 把深度也进行混合
#ifdef LOGFORWARDTIME
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &RasterizeToPixels_start);
#endif
    auto rasterize_res = RasterizeToPixelsGes_NewParallel::apply(means2d,
                                                                 conics,
                                                                 colors,
                                                                 getRealOpacities(),
                                                                 radiis,
                                                                 ref_depth_clamped,
                                                                 base_color,
                                                                 backgrounds,
                                                                 tile_masks,
                                                                 cam.width,
                                                                 cam.height,
                                                                 tile_size,
                                                                 isect_offsets,
                                                                 flatten_ids,
                                                                 group_gs_ids,
                                                                 group_starts,
                                                                 abs_grad,
                                                                 delta_depth);

    torch::Tensor render_colors = rasterize_res[0];
    torch::Tensor weight_sums = rasterize_res[1];

#ifdef LOGFORWARDTIME
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &RasterizeToPixels_end);
#endif
    // STEP5: 计算加权平均深度
    int64_t last_dim = render_colors.size(-1);
    torch::Tensor raw_rgb = render_colors.slice(-1, 0, last_dim - 1);
    torch::Tensor raw_depth = render_colors.slice(-1, last_dim - 1);
    auto base_color_wight = 1.0f * torch::ones_like(weight_sums);
    // base_color_wight.masked_fill_(ref_depth > 0, 1);
    torch::Tensor expected_rgb = (raw_rgb + base_color * base_color_wight) / (weight_sums + base_color_wight);
    auto base_depth_wight = 1.0f * torch::zeros_like(weight_sums);
    base_depth_wight.masked_fill_(ref_depth > 0, 1);
    torch::Tensor expected_depth = (raw_depth + ref_depth * base_depth_wight) / (weight_sums + base_depth_wight);

    // rgb = torch::clamp_max(rgb, 1.0f);
    TensorDict res;
    // all render res with shape [N, H, W, C], and we assumue one camera!
    if (!use_exposure)
        res["rgb"] = expected_rgb[0];
    else
    {
        if (cam.id >= getExposure().size(0))
        {
            res["rgb"] = expected_rgb[0];
        }
        else
        {
            torch::Tensor exposure = getExposure()[cam.id];
            auto linear_transform = exposure.slice(1, 0, 3).t();
            auto offset = exposure.slice(1, 3, 4).t();
            res["rgb"] = torch::matmul(expected_rgb[0], linear_transform) + offset;
        }
    }
    res["depth"] = expected_depth[0];
    res["alpha"] = weight_sums[0];
    res["radiis"] = radiis[0];
    res["means2d"] = means2d; // 这里必须返回原tensor，切片操作会导致无法记录梯度
#ifdef LOGFORWARDTIME
    auto proj_time = calculateTimeInterval(FullyFusedProjection_start, FullyFusedProjection_end);
    auto sh_time = calculateTimeInterval(SphericalHarmonics_start, SphericalHarmonics_end);
    auto tile_insect_time = calculateTimeInterval(isectTiles_start, isectTiles_end);
    auto rasterize_time = calculateTimeInterval(RasterizeToPixels_start, RasterizeToPixels_end);
    auto forward_time = calculateTimeInterval(FullyFusedProjection_start, RasterizeToPixels_end);
    printf("proj_time: %f, sh_time: %f, tile_insect_time: %f, rasterize_time: %f, forward_time: %f, max_radii: %d\n",
           proj_time, sh_time, tile_insect_time, rasterize_time, forward_time, res["radiis"].max().item<int>());
    res["opt_state/proj_time"] = torch::tensor(proj_time);
    res["opt_state/sh_time"] = torch::tensor(sh_time);
    res["opt_state/tile_insect_time"] = torch::tensor(tile_insect_time);
    res["opt_state/rasterize_time"] = torch::tensor(rasterize_time);
    res["opt_state/forward_time"] = torch::tensor(forward_time);
#endif

    return res;
}

TensorDict RawGaussianModel::computeLoss(TensorDict &render_res, const Camera &cam, const YAML::Node &weight_configs, const torch::Tensor &mask)
{
    torch::Tensor gt_rgb = cam.image.to(device);
    torch::Tensor gt_depth = cam.depth.to(device);
    torch::Tensor rendered_rgb = render_res["rgb"];
    torch::Tensor rendered_depth = render_res["depth"];
    torch::Tensor rendered_weight_sum = render_res["alpha"];
    // torch::Tensor valid_depth_mask = gt_depth > 0.2 & gt_depth < 3;
    torch::Tensor rgb_loss, l1_loss;
    if (mask.defined())
        l1_loss = l1(gt_rgb.masked_select(mask), rendered_rgb.masked_select(mask));
    else
        l1_loss = l1(gt_rgb, rendered_rgb);

    // torch::Tensor l1_loss = l1(gt_rgb.masked_select(valid_depth_mask), rendered_rgb.masked_select(valid_depth_mask));

    float ssimWeight = weight_configs["ssim_weight"].as<float>();
    if (ssimWeight > 0)
    {
        float C1 = 0.01 * 0.01;
        float C2 = 0.03 * 0.03;
        std::string padding = "valid";
        torch::Tensor ssimLoss = 1.0f - FusedSSIMMap::apply(C1,
                                                            C2,
                                                            rendered_rgb.permute({2, 0, 1}).unsqueeze(0),
                                                            gt_rgb.permute({2, 0, 1}).unsqueeze(0),
                                                            padding,
                                                            true)
                                            .mean();
        rgb_loss = (1.0f - ssimWeight) * l1_loss + ssimWeight * ssimLoss;
    }
    else
        rgb_loss = l1_loss;

    float depth_weight = weight_configs["depth_weight"].as<float>();
    TensorDict loss;
    loss["total"] = rgb_loss;
    loss["rgb"] = rgb_loss;

    if (depth_weight > 0 && cam.has_depth)
    {
        auto valid_depth_mask = (gt_depth > 0) & (rendered_depth > 0);
        torch::Tensor depthLoss = l1(gt_depth.masked_select(valid_depth_mask), rendered_depth.masked_select(valid_depth_mask));
        loss["depth"] = depthLoss;
        loss["total"] += depth_weight * depthLoss;
    }

    return loss;
}

void RawGaussianModel::stepPostBackward(TensorDict &render_res, const Camera &cam, float scene_scale, int curr_iter)
{
    torch::NoGradGuard noGrad;
    // densify parameters
    int densify_interval = config["densify_interval"].as<int>();
    int densify_start_iter = config["densify_start_iter"].as<int>();
    int densify_end_iter = config["densify_end_iter"].as<int>();
    int reset_opacity_interval = config["reset_opacity_interval"].as<int>();
    float reset_opacity_target = 2 * config["prune_opacity_thres"].as<float>();
    if (curr_iter < densify_end_iter)
    {
        // 1. 记录高斯梯度
        updateDensifyGrad(render_res, cam);
        if (curr_iter % densify_interval == 0 && curr_iter > densify_start_iter)
        {
            // 2. 进行高斯分裂/克隆/删除操作
            std::cout << "gs num before densification: " << getGaussianNum() << std::endl;
            densifiyGs(scene_scale, curr_iter);
            std::cout << "gs num after densification: " << getGaussianNum() << std::endl;

            // 需要删除之前记录过的信息，否则第二次Densification时用的还是上一次的
            grad_2d = torch::Tensor();
            visible_count = torch::Tensor();
            max2DSize = torch::Tensor();
            // std::cout << "mean opacities: " << opt_gs_params.getRealOpacities().mean().item<float>() << std::endl;
            // torch::cuda::empty_cache();
        }
        // 3. 重置透明度
        if (curr_iter % reset_opacity_interval == 0)
        {
            torch::Tensor new_opacities = torch::clamp_max(opt_gs_params.opacities, torch::logit(torch::tensor(reset_opacity_target)).item<float>()).detach().requires_grad_().to(device);
            // checkOpimizer(opacitiesOpt);
            replaceToOptimizer(opacitiesOpt, new_opacities);
            opt_gs_params.opacities = new_opacities;
            std::cout << "Alpha reset: " << reset_opacity_target << std::endl;
        }
    }
}

// 更新记录的梯度
void RawGaussianModel::updateDensifyGrad(TensorDict &render_res, const Camera &cam)
{
    // 获取记录高斯梯度
    torch::NoGradGuard no_grad;
    torch::Tensor visibleMask = (render_res["radiis"] > 0).flatten();
    torch::Tensor raw_grads = render_res["means2d"].grad().detach().clone();
    // raw_grads可能是C, N, 2
    if (raw_grads.dim() == 3 && raw_grads.sizes()[0] == 1)
    {
        raw_grads = raw_grads.squeeze(0);
    }

    auto img_size = torch::tensor({lastWidth, lastHeight}, raw_grads.options());
    auto scaled_grads = raw_grads * img_size / 2;
    torch::Tensor grads_norm = scaled_grads.norm(2, -1);

    if (!grad_2d.numel())
    {
        grad_2d = torch::zeros_like(grads_norm).to(device);
        visible_count = torch::zeros_like(grads_norm).to(device);
        std::cout << "zero state" << std::endl;
    }
    visible_count.index_put_({visibleMask}, visible_count.index({visibleMask}) + 1.0f);
    grad_2d.index_put_({visibleMask}, grads_norm.index({visibleMask}) + grad_2d.index({visibleMask}));

    // std::cout << "raw grad: " << raw_grads.mean() << std::endl;
    // // std::cout << "visibale mask: " << visibleMask.sum() << std::endl;
    // // std::cout << "scaled grad: " << scaled_grads.mean() << std::endl;
    // std::cout << "grads norm: " << grads_norm.index({visibleMask}).mean() << std::endl;
    // std::cout << "logged grad: " << grad_2d.mean() << std::endl;
    // std::cout << "logged mask: " << visible_count.mean() << std::endl;

    // max2DSize是记录scale大于一定屏幕像素尺寸的高斯
    if (!max2DSize.numel())
    {
        max2DSize = torch::zeros_like(render_res["radiis"], torch::kFloat32);
    }
    // NOTE: 同样这里使用图像尺寸进行归一化的操作原版也没有
    torch::Tensor newRadii = render_res["radiis"].detach().index({visibleMask});
    max2DSize.index_put_({visibleMask}, torch::maximum(
                                            max2DSize.index({visibleMask}), newRadii));
}

void RawGaussianModel::densifiyGs(float scene_scale, int curr_iter)
{
    torch::NoGradGuard no_grad;
    int densify_interval = config["densify_interval"].as<int>();
    float grad_thres = config["densify_grad_thres"].as<float>();
    float large_thres = config["densify_large_thres"].as<float>();
    float split_screen_size = config["split_screen_size"].as<float>();
    float prune_opacity_thres = config["prune_opacity_thres"].as<float>();
    int reset_opacity_interval = config["reset_opacity_interval"].as<int>();
    std::cout << "mean opac: " << getRealOpacities().mean().item() << std::endl;
    // NOTE: 这个判断是opensplat的实现，也就是重置opacity后要等一段时间稳定后再进行densify
    if (curr_iter % reset_opacity_interval >= pause_refine_after_reset)
    {

        // STEP1. 计算平均梯度
        torch::Tensor grads = grad_2d / torch::clamp_min(visible_count, 1);
        std::cout << "grads mean: " << grads.mean().item() << std::endl;
        torch::Tensor is_grad_high = grads > grad_thres;
        // std::cout << grads.slice(0, 0, 10) << std::endl;
        // std::cout << grad_thres << std::endl;
        std::cout << "grads high num: " << is_grad_high.sum().item() << std::endl;
        std::cout << "grads high ratio: " << (is_grad_high.sum() / getGaussianNum()).item() << std::endl;
        torch::Tensor max_scales = std::get<0>(torch::max(getRealScales(), 1));
        torch::Tensor is_scale_large = max_scales > large_thres * scene_scale;
        auto is_dupli = is_grad_high & (~is_scale_large);
        auto is_split = is_grad_high & is_scale_large;
        // is_split.zero_();
        std::cout << "dupli num: " << is_dupli.sum().item() << std::endl;
        std::cout << "split num: " << is_split.sum().item() << std::endl;

        // STEP2. 对尺寸较小并且梯度较大的高斯进行克隆
        torch::Tensor dupMeans = opt_gs_params.getMeans().index({is_dupli});
        torch::Tensor dupFeaturesDc = opt_gs_params.getFeaturesDc().index({is_dupli});
        torch::Tensor dupFeaturesRest = opt_gs_params.getFeaturesRest().index({is_dupli});
        torch::Tensor dupOpacities = opt_gs_params.getOpacities().index({is_dupli});
        torch::Tensor dupScales = opt_gs_params.getScales().index({is_dupli});
        torch::Tensor dupQuats = opt_gs_params.getQuats().index({is_dupli});

        // STEP3. 对梯度较大，尺寸较大的高斯进行分裂
        const int nSplitSamples = 2; // 分裂的个数
        int nSplits = is_split.sum().item<int>();

        torch::Tensor centeredSamples = torch::randn({nSplitSamples * nSplits, 3}, device); // Nx3 of axis-aligned scales
        torch::Tensor scaledSamples = torch::exp(opt_gs_params.scales.index({is_split}).repeat({nSplitSamples, 1})) * centeredSamples;
        torch::Tensor qs = opt_gs_params.quats.index({is_split}) / torch::linalg_vector_norm(opt_gs_params.quats.index({is_split}), 2, {-1}, true, torch::kFloat32);
        torch::Tensor rots = quatToRotMat(qs.repeat({nSplitSamples, 1}));
        torch::Tensor rotatedSamples = torch::bmm(rots, scaledSamples.index({"...", None})).squeeze();
        torch::Tensor splitMeans = (rotatedSamples + opt_gs_params.means.index({is_split}).repeat({nSplitSamples, 1}));
        torch::Tensor splitFeaturesDc = opt_gs_params.featuresDc.index({is_split}).repeat({nSplitSamples, 1});
        torch::Tensor splitFeaturesRest = opt_gs_params.featuresRest.index({is_split}).repeat({nSplitSamples, 1, 1});
        torch::Tensor splitOpacities = opt_gs_params.opacities.index({is_split}).repeat({nSplitSamples, 1});
        const float sizeFac = 1.6f;
        torch::Tensor splitScales = torch::log(torch::exp(opt_gs_params.scales.index({is_split})) / sizeFac).repeat({nSplitSamples, 1});
        opt_gs_params.scales.index({is_split}) = torch::log(torch::exp(opt_gs_params.scales.index({is_split})) / sizeFac);
        torch::Tensor splitQuats = opt_gs_params.getQuats().index({is_split}).repeat({nSplitSamples, 1});

        opt_gs_params.means = torch::cat({opt_gs_params.means.detach(), splitMeans, dupMeans}, 0).requires_grad_();
        opt_gs_params.featuresDc = torch::cat({opt_gs_params.featuresDc.detach(), splitFeaturesDc, dupFeaturesDc}, 0).requires_grad_();
        opt_gs_params.featuresRest = torch::cat({opt_gs_params.featuresRest.detach(), splitFeaturesRest, dupFeaturesRest}, 0).requires_grad_();
        opt_gs_params.opacities = torch::cat({opt_gs_params.opacities.detach(), splitOpacities, dupOpacities}, 0).requires_grad_();
        opt_gs_params.scales = torch::cat({opt_gs_params.scales.detach(), splitScales, dupScales}, 0).requires_grad_();
        opt_gs_params.quats = torch::cat({opt_gs_params.quats.detach(), splitQuats, dupQuats}, 0).requires_grad_();

        max2DSize = torch::cat({max2DSize,
                                torch::zeros_like(splitScales.index({Slice(), 0})),
                                torch::zeros_like(dupScales.index({Slice(), 0}))},
                               0);

        torch::Tensor dupIdcs = torch::where(is_dupli)[0];
        addToOptimizer(meansOpt, opt_gs_params.means, dupIdcs, 1);
        addToOptimizer(scalesOpt, opt_gs_params.scales, dupIdcs, 1);
        addToOptimizer(quatsOpt, opt_gs_params.quats, dupIdcs, 1);
        addToOptimizer(featuresDcOpt, opt_gs_params.featuresDc, dupIdcs, 1);
        addToOptimizer(featuresRestOpt, opt_gs_params.featuresRest, dupIdcs, 1);
        addToOptimizer(opacitiesOpt, opt_gs_params.opacities, dupIdcs, 1);

        torch::Tensor splitIdcs = torch::where(is_split)[0];
        addToOptimizer(meansOpt, opt_gs_params.means, splitIdcs, nSplitSamples);
        addToOptimizer(scalesOpt, opt_gs_params.scales, splitIdcs, nSplitSamples);
        addToOptimizer(quatsOpt, opt_gs_params.quats, splitIdcs, nSplitSamples);
        addToOptimizer(featuresDcOpt, opt_gs_params.featuresDc, splitIdcs, nSplitSamples);
        addToOptimizer(featuresRestOpt, opt_gs_params.featuresRest, splitIdcs, nSplitSamples);
        addToOptimizer(opacitiesOpt, opt_gs_params.opacities, splitIdcs, nSplitSamples);

        // 这里要做这个更新mask的操作是因为后面在prune的时候要把split之前的大尺寸高斯全都删掉，要正确地更新tensor的尺寸
        torch::Tensor splitsMask = torch::cat({is_split,
                                               torch::full({nSplitSamples * is_split.sum().item<int>() + is_dupli.sum().item<int>()}, false, torch::TensorOptions().dtype(torch::kBool).device(device))},
                                              0);
        // STEP4. Prune操作，包括删除透明度过低的高斯，删除尺寸过大的高斯，删除屏幕尺寸过大的高斯
        int numPointsBefore = getGaussianNum();
        torch::Tensor all_opac = opt_gs_params.getRealOpacities();

        torch::Tensor is_prune = (all_opac < prune_opacity_thres).squeeze();
        std::cout << "prune opac low num: " << is_prune.sum().item() << std::endl;
        if (splitsMask.numel())
        {
            is_prune |= splitsMask;
        }

        if (curr_iter > reset_opacity_interval)
        {
            const float pruneScaleThresh = 0.1f; // cull huge gaussians
            // const float pruneScreenSize = 0.15;
            const float pruneScreenSize = 20.f; // % of screen space
            torch::Tensor is_large_scale = std::get<0>(torch::exp(opt_gs_params.getScales()).max(-1)) > pruneScaleThresh * scene_scale;
            std::cout << "prune scale size gs: " << is_large_scale.sum().item() << std::endl;
            // std::cout << "prune screen size gs: " << (max2DSize > pruneScreenSize).sum().item() << std::endl;
            // is_large_scale |= max2DSize > pruneScreenSize;
            is_prune |= is_large_scale;
        }

        int prune_count = torch::sum(is_prune).item<int>();
        if (prune_count > 0)
        {
            opt_gs_params.means = opt_gs_params.getMeans().index({~is_prune}).detach().requires_grad_();
            opt_gs_params.scales = opt_gs_params.getScales().index({~is_prune}).detach().requires_grad_();
            opt_gs_params.quats = opt_gs_params.getQuats().index({~is_prune}).detach().requires_grad_();
            opt_gs_params.featuresDc = opt_gs_params.getFeaturesDc().index({~is_prune}).detach().requires_grad_();
            opt_gs_params.featuresRest = opt_gs_params.getFeaturesRest().index({~is_prune}).detach().requires_grad_();
            opt_gs_params.opacities = opt_gs_params.getOpacities().index({~is_prune}).detach().requires_grad_();

            removeFromOptimizer(meansOpt, opt_gs_params.means, is_prune);
            removeFromOptimizer(scalesOpt, opt_gs_params.scales, is_prune);
            removeFromOptimizer(quatsOpt, opt_gs_params.quats, is_prune);
            removeFromOptimizer(featuresDcOpt, opt_gs_params.featuresDc, is_prune);
            removeFromOptimizer(featuresRestOpt, opt_gs_params.featuresRest, is_prune);
            removeFromOptimizer(opacitiesOpt, opt_gs_params.opacities, is_prune);

            // std::cout << "Pruned " << (numPointsBefore - opt_gs_params.getGaussianNum()) << " gaussians, remaining " << opt_gs_params.getGaussianNum() << std::endl;
        }
    }
}

void RawGaussianModel::prunePoints(const torch::Tensor &deleteMask)
{
    opt_gs_params.remove(deleteMask);
    removeFromOptimizer(meansOpt, opt_gs_params.means, deleteMask);
    removeFromOptimizer(scalesOpt, opt_gs_params.scales, deleteMask);
    removeFromOptimizer(quatsOpt, opt_gs_params.quats, deleteMask);
    removeFromOptimizer(featuresDcOpt, opt_gs_params.featuresDc, deleteMask);
    removeFromOptimizer(featuresRestOpt, opt_gs_params.featuresRest, deleteMask);
    removeFromOptimizer(opacitiesOpt, opt_gs_params.opacities, deleteMask);
}

// 目前此函数只读取了模型参数没有读取优化器参数，只支持从头训练
void RawGaussianModel::loadParamsTensor(const std::string &filename)
{
    // getGaussianParms().loadTensor(filename);
    opt_gs_params.loadTensor(filename);
}

////////////////////  Optimizer Functions ////////////////////
void RawGaussianModel::initOptimizers(int max_iterations, float scene_scale)
{
    if (meansOpt != nullptr)
    { // 每次运行都会重置所有优化器
        delete meansOpt, delete scalesOpt, delete quatsOpt, delete featuresDcOpt, delete featuresRestOpt, delete opacitiesOpt, delete meansOptScheduler;
        meansOpt = nullptr, scalesOpt = nullptr, quatsOpt = nullptr, featuresDcOpt = nullptr, featuresRestOpt = nullptr, opacitiesOpt = nullptr, meansOptScheduler = nullptr, exposureOpt = nullptr;
    }
    float BS = 1;
    float eps = 1e-15 / std::sqrt(BS);
    float B1 = 1 - BS * (1 - 0.9);
    float B2 = 1 - BS * (1 - 0.999);

    meansOpt = new torch::optim::Adam({opt_gs_params.means}, torch::optim::AdamOptions(meansLr * scene_scale).eps(eps).betas(std::make_tuple(B1, B2)));
    scalesOpt = new torch::optim::Adam({opt_gs_params.scales}, torch::optim::AdamOptions(scalesLr).eps(eps).betas(std::make_tuple(B1, B2)));
    quatsOpt = new torch::optim::Adam({opt_gs_params.quats}, torch::optim::AdamOptions(quatsLr).eps(eps).betas(std::make_tuple(B1, B2)));
    featuresDcOpt = new torch::optim::Adam({opt_gs_params.featuresDc}, torch::optim::AdamOptions(featuresDcLr).eps(eps).betas(std::make_tuple(B1, B2)));
    featuresRestOpt = new torch::optim::Adam({opt_gs_params.featuresRest}, torch::optim::AdamOptions(featuresRestLr).eps(eps).betas(std::make_tuple(B1, B2)));
    opacitiesOpt = new torch::optim::Adam({opt_gs_params.opacities}, torch::optim::AdamOptions(opacitiesLr).eps(eps).betas(std::make_tuple(B1, B2)));
    exposureOpt = new torch::optim::Adam({opt_gs_params.exposure}, torch::optim::AdamOptions(exposureLr).eps(eps).betas(std::make_tuple(B1, B2)));
    if (max_iterations > 0)
        meansOptScheduler = new OptimScheduler(meansOpt, std::pow(0.01, 1.0f / max_iterations));
}

void RawGaussianModel::optimizersZeroGrad()
{
    meansOpt->zero_grad();
    scalesOpt->zero_grad();
    quatsOpt->zero_grad();
    featuresDcOpt->zero_grad();
    featuresRestOpt->zero_grad();
    opacitiesOpt->zero_grad();
    exposureOpt->zero_grad();
    opt_gs_params.means.mutable_grad().reset();
    opt_gs_params.scales.mutable_grad().reset();
    opt_gs_params.quats.mutable_grad().reset();
    opt_gs_params.featuresDc.mutable_grad().reset();
    opt_gs_params.featuresRest.mutable_grad().reset();
    opt_gs_params.opacities.mutable_grad().reset();
    opt_gs_params.exposure.mutable_grad().reset();
}

// 根据优化器更新参数
void RawGaussianModel::optimizersStep()
{
    meansOpt->step();
    scalesOpt->step();
    quatsOpt->step();
    featuresDcOpt->step();
    featuresRestOpt->step();
    opacitiesOpt->step();
    exposureOpt->step();
}

void RawGaussianModel::addToOptimizer(torch::optim::Adam *optimizer, const torch::Tensor &newParam, const torch::Tensor &idcs, int nSamples)
{
    torch::Tensor param = optimizer->param_groups()[0].params()[0];
#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR > 1
    auto pId = param.unsafeGetTensorImpl();
#else
    auto pId = c10::guts::to_string(param.unsafeGetTensorImpl());
#endif
    auto paramState = std::make_unique<torch::optim::AdamParamState>(static_cast<torch::optim::AdamParamState &>(*optimizer->state()[pId]));

    std::vector<int64_t> repeats;
    repeats.push_back(nSamples);
    for (long int i = 0; i < paramState->exp_avg().dim() - 1; i++)
    {
        repeats.push_back(1);
    }

    paramState->exp_avg(torch::cat({paramState->exp_avg(),
                                    torch::zeros_like(paramState->exp_avg().index({idcs.squeeze()})).repeat(repeats)},
                                   0));

    paramState->exp_avg_sq(torch::cat({paramState->exp_avg_sq(),
                                       torch::zeros_like(paramState->exp_avg_sq().index({idcs.squeeze()})).repeat(repeats)},
                                      0));

    optimizer->state().erase(pId);

#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR > 1
    auto newPId = newParam.unsafeGetTensorImpl();
#else
    auto newPId = c10::guts::to_string(newParam.unsafeGetTensorImpl());
#endif
    optimizer->state()[newPId] = std::move(paramState);

    optimizer->param_groups()[0].params()[0] = newParam;
}

void RawGaussianModel::removeFromOptimizer(torch::optim::Adam *optimizer, const torch::Tensor &newParam, const torch::Tensor &deletedMask)
{
    torch::Tensor param = optimizer->param_groups()[0].params()[0];
#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR > 1
    auto pId = param.unsafeGetTensorImpl();
#else
    auto pId = c10::guts::to_string(param.unsafeGetTensorImpl());
#endif
    auto paramState = std::make_unique<torch::optim::AdamParamState>(static_cast<torch::optim::AdamParamState &>(*optimizer->state()[pId]));

    paramState->exp_avg(paramState->exp_avg().index({~deletedMask}));
    paramState->exp_avg_sq(paramState->exp_avg_sq().index({~deletedMask}));

    optimizer->state().erase(pId);
#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR > 1
    auto newPId = newParam.unsafeGetTensorImpl();
#else
    auto newPId = c10::guts::to_string(newParam.unsafeGetTensorImpl());
#endif
    optimizer->param_groups()[0].params()[0] = newParam;
    optimizer->state()[newPId] = std::move(paramState);
}

void RawGaussianModel::replaceToOptimizer(torch::optim::Adam *optimizer, const torch::Tensor &newParam)
{
    torch::Tensor param = optimizer->param_groups()[0].params()[0];
#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR > 1
    auto pId = param.unsafeGetTensorImpl();
#else
    auto pId = c10::guts::to_string(param.unsafeGetTensorImpl());
#endif
    // 检查state是否存在，因为在第一个iteration时没有state，无需操作
    bool has_state = false;
    if (optimizer->state().find(pId) != optimizer->state().end())
    {
        auto &state = optimizer->state()[pId];
        if (auto *adamState = dynamic_cast<torch::optim::AdamParamState *>(state.get()))
        {
            // 现在可以安全地使用 adamState
            has_state = true;
        }
        else
        {
            std::cout << "State is not AdamParamState" << std::endl;
        }
    }
    else
    {
        auto init_state = std::make_unique<torch::optim::AdamParamState>();
        // // 初始化 step
        init_state->step() = 0;
        // 初始化 exp_avg
        init_state->exp_avg(torch::zeros_like(param, param.options()));

        // 初始化 exp_avg_sq
        init_state->exp_avg_sq(torch::zeros_like(param, param.options()));
        // // 如果使用 AMSGrad，还需要初始化 max_exp_avg_sq
        // if (optimizer->options.amsgrad())
        // {
        //     init_state->max_exp_avg_sq(torch::zeros_like(param, param.options()));
        // }
        optimizer->state()[pId] = std::move(init_state);
        std::cout << "No state found for this parameter!!!" << std::endl;
    }

    // 把state重置为0
    auto paramState = std::make_unique<torch::optim::AdamParamState>(static_cast<torch::optim::AdamParamState &>(*optimizer->state()[pId]));
    paramState->exp_avg(torch::zeros_like(paramState->exp_avg()));
    paramState->exp_avg_sq(torch::zeros_like(paramState->exp_avg_sq()));

    optimizer->state().erase(pId);
#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR > 1
    auto newPId = newParam.unsafeGetTensorImpl();
#else
    auto newPId = c10::guts::to_string(newParam.unsafeGetTensorImpl());
#endif
    optimizer->state()[newPId] = std::move(paramState);

    optimizer->param_groups()[0].params()[0] = newParam;
}