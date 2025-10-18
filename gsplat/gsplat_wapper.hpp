#ifndef GSPLAT_NEW_WAPPER_H
#define GSPLAT_NEW_WAPPER_H

#include "rasterizer/bindings.h"
#include "rasterizer/ssim.h"
#include "rasterizer/simple_knn.h"


#include <torch/torch.h>
using namespace torch::autograd;

// #define LOGBACKWARDTIME

double getDuration(struct timespec start, struct timespec end);

class SphericalHarmonicsNew : public Function<SphericalHarmonicsNew>
{
public:
    static torch::Tensor forward(AutogradContext *ctx,
                                 int sh_degree,
                                 torch::Tensor dirs,
                                 torch::Tensor coeffs,
                                 torch::Tensor masks)
    {
        long long numPoints = coeffs.size(0);

        ctx->save_for_backward({dirs, coeffs, masks});
        ctx->saved_data["sh_degree"] = sh_degree;
        ctx->saved_data["num_bases"] = coeffs.size(-2);

        // 在python中这里用的是ext.cpp定义的函数名，在我们这里直接使用本来的函数名
        // return compute_sh_fwd(sh_degree, dirs, coeffs, masks);
        return gsplat::compute_sh_fwd_tensor(sh_degree, dirs, coeffs, masks);
    }
    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs)
    {
#ifdef LOGBACKWARDTIME
        struct timespec sh_bwd_start, sh_bwd_end, kernel_start, kernel_end;
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &sh_bwd_start);
#endif

        auto saved = ctx->get_saved_variables();
        auto dirs = saved[0];
        auto coeffs = saved[1];
        auto masks = saved[2];

        int sh_degree = ctx->saved_data["sh_degree"].toInt();
        int num_bases = ctx->saved_data["num_bases"].toInt();

        auto v_colors = grad_outputs[0].contiguous();
        bool compute_v_dirs = ctx->needs_input_grad(1);

        // auto [v_coeffs, v_dirs] = compute_sh_bwd(num_bases, sh_degree, dirs, coeffs, masks, v_colors, compute_v_dirs);

        // torch::Tensor v_coeffs = torch::zeros_like(coeffs);
        // torch::Tensor v_dirs;

#ifdef LOGBACKWARDTIME
        std::cout << "dirs: " << dirs.sizes() << std::endl;
        std::cout << "coeffs: " << coeffs.sizes() << std::endl;
        std::cout << "compute_v_dirs: " << compute_v_dirs << std::endl;
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &kernel_start);
#endif
        // gsplat::compute_sh_bwd_tensor_new(num_bases, sh_degree, dirs, coeffs, masks, v_colors, compute_v_dirs, v_coeffs, v_dirs);
        auto result = gsplat::compute_sh_bwd_tensor(num_bases, sh_degree, dirs, coeffs, masks, v_colors, compute_v_dirs);
#ifdef LOGBACKWARDTIME

        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &kernel_end);
#endif
        torch::Tensor v_coeffs = std::get<0>(result);
        torch::Tensor v_dirs = std::get<1>(result);

        // std::cout << "SphericalHarmonicsNew backward" << std::endl;
        // std::cout << v_colors.mean() << std::endl;
        // std::cout << v_coeffs.mean() << std::endl;
        // std::cout << v_coeffs.mean() << std::endl;

        torch::autograd::tensor_list grad_inputs;
        grad_inputs.push_back(torch::Tensor()); // None for sh_degree
        grad_inputs.push_back(compute_v_dirs ? v_dirs : torch::Tensor());
        grad_inputs.push_back(v_coeffs);
        grad_inputs.push_back(torch::Tensor()); // None for masks
#ifdef LOGBACKWARDTIME
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &sh_bwd_end);
        auto sh_bwd_time = getDuration(sh_bwd_start, sh_bwd_end);
        auto kernel_time = getDuration(kernel_start, kernel_end);
        std::cout << "SH bwd time: " << sh_bwd_time << "\tkernel time: " << kernel_time << std::endl;
#endif
        return grad_inputs;
    }
};

class FullyFusedProjection : public Function<FullyFusedProjection>
{
public:
    static variable_list forward(
        AutogradContext *ctx,
        torch::Tensor means,
        at::optional<torch::Tensor> covars,
        torch::Tensor quats,
        torch::Tensor scales,
        torch::Tensor viewmats,
        torch::Tensor Ks,
        int width,
        int height,
        float eps2d,
        float near_plane,
        float far_plane,
        float radius_clip,
        bool calc_compensations,
        std::string camera_model)
    {
        means = means.contiguous();
        quats = quats.contiguous();
        scales = scales.contiguous();
        viewmats = viewmats.contiguous();

        gsplat::CameraModelType camera_model_type;
        if (camera_model == "pinhole")
            camera_model_type = gsplat::PINHOLE;
        else if (camera_model == "ortho")
            camera_model_type = gsplat::ORTHO;
        else if (camera_model == "fisheye")
            camera_model_type = gsplat::FISHEYE;
        else
            throw std::runtime_error("Unknown camera model");

        auto result = gsplat::fully_fused_projection_fwd_tensor(
            means, covars, quats, scales, viewmats, Ks,
            width, height, eps2d, near_plane, far_plane, radius_clip,
            calc_compensations, camera_model_type);

        auto radii = std::get<0>(result);
        auto means2d = std::get<1>(result);
        auto depths = std::get<2>(result);
        auto conics = std::get<3>(result);
        auto compensations = std::get<4>(result);

        if (!calc_compensations)
        {
            compensations = torch::zeros({1}, means.options());
        }
        ctx->save_for_backward({means, quats, scales, viewmats, Ks, radii, conics});

        ctx->saved_data["width"] = width;
        ctx->saved_data["height"] = height;
        ctx->saved_data["eps2d"] = eps2d;
        ctx->saved_data["camera_model_type"] = camera_model_type;
        return {radii, means2d, depths, conics, compensations};
    }

    static variable_list backward(
        AutogradContext *ctx,
        variable_list grad_outputs)
    {
#ifdef LOGBACKWARDTIME
        struct timespec proj_bwd_start, proj_bwd_end;
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &proj_bwd_start);
#endif
        auto saved = ctx->get_saved_variables();
        auto means = saved[0];
        auto quats = saved[1];
        auto scales = saved[2];
        auto viewmats = saved[3];
        auto Ks = saved[4];
        auto radii = saved[5];
        auto conics = saved[6];

        at::optional<torch::Tensor> covars, compensations;

        int width = ctx->saved_data["width"].toInt();
        int height = ctx->saved_data["height"].toInt();
        float eps2d = ctx->saved_data["eps2d"].toDouble();
        int camera_model_type_int = ctx->saved_data["camera_model_type"].toInt();

        gsplat::CameraModelType camera_model_type;
        if (camera_model_type_int == 0)
            camera_model_type = gsplat::PINHOLE;
        else if (camera_model_type_int == 1)
            camera_model_type = gsplat::ORTHO;
        else if (camera_model_type_int == 2)
            camera_model_type = gsplat::FISHEYE;

        auto v_radii = grad_outputs[0];
        auto v_means2d = grad_outputs[1];
        auto v_depths = grad_outputs[2];
        auto v_conics = grad_outputs[3];
        at::optional<torch::Tensor> v_compensations;

        auto result = gsplat::fully_fused_projection_bwd_tensor(
            means, covars, quats, scales, viewmats, Ks,
            width, height, eps2d, camera_model_type,
            radii, conics, compensations,
            v_means2d.contiguous(), v_depths.contiguous(), v_conics.contiguous(),
            v_compensations, ctx->needs_input_grad(4));

        auto v_means = std::get<0>(result);
        auto v_covars = std::get<1>(result);
        auto v_quats = std::get<2>(result);
        auto v_scales = std::get<3>(result);
        auto v_viewmats = std::get<4>(result);

        // 手动指定不需要计算梯度的tensor
        v_covars = torch::Tensor();
        v_viewmats = torch::Tensor();
// if (!ctx->needs_input_grad(0))
//     v_means = torch::Tensor();
// if (!ctx->needs_input_grad(1))
//     v_covars = torch::Tensor();
// if (!ctx->needs_input_grad(2))
//     v_quats = torch::Tensor();
// if (!ctx->needs_input_grad(3))
//     v_scales = torch::Tensor();
// if (!ctx->needs_input_grad(4))
//     v_viewmats = torch::Tensor();
// std::cout << "FullyFusedProjection backward input" << std::endl;
// std::cout << v_means2d.mean() << std::endl;
// std::cout << v_conics.mean() << std::endl;

// std::cout << v_means.mean() << std::endl;
// std::cout << v_quats.mean() << std::endl;
// std::cout << v_scales.mean() << std::endl;
// std::cout << "FullyFusedProjection backward finish" << std::endl;
#ifdef LOGBACKWARDTIME
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &proj_bwd_end);
        auto proj_bwd_time = getDuration(proj_bwd_start, proj_bwd_end);
        std::cout << "Proj bwd time: " << proj_bwd_time << std::endl;
#endif
        return {
            v_means, v_covars, v_quats, v_scales, v_viewmats,
            torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
            torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
            torch::Tensor()};
    }
};

class RasterizeToPixels : public Function<RasterizeToPixels>
{
public:
    static variable_list forward(
        AutogradContext *ctx,
        torch::Tensor means2d,
        torch::Tensor conics,
        torch::Tensor colors,
        torch::Tensor opacities,
        at::optional<torch::Tensor> backgrounds,
        at::optional<torch::Tensor> masks,
        int width,
        int height,
        int tile_size,
        torch::Tensor isect_offsets,
        torch::Tensor flatten_ids,
        bool absgrad)
    {
        if (backgrounds.has_value())
            backgrounds = backgrounds->contiguous();
        if (masks.has_value())
            masks = masks->contiguous();

        auto result = gsplat::rasterize_to_pixels_fwd_tensor(
            means2d.contiguous(), conics.contiguous(), colors.contiguous(), opacities.contiguous(), backgrounds, masks,
            width, height, tile_size, isect_offsets.contiguous(), flatten_ids.contiguous());

        auto render_colors = std::get<0>(result);
        auto render_alphas = std::get<1>(result);
        auto last_ids = std::get<2>(result);

        ctx->save_for_backward({means2d, conics, colors, opacities,
                                isect_offsets, flatten_ids, render_alphas, last_ids});
        ctx->saved_data["width"] = width;
        ctx->saved_data["height"] = height;
        ctx->saved_data["tile_size"] = tile_size;
        ctx->saved_data["absgrad"] = absgrad;
        // ctx->saved_data["backgronds"] = backgrounds;
        // ctx->saved_data["masks"] = masks;
        return {render_colors, render_alphas};
    }

    static variable_list backward(
        AutogradContext *ctx,
        variable_list grad_outputs)
    {
#ifdef LOGBACKWARDTIME
        struct timespec rasterize_bwd_start, rasterize_bwd_end;
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &rasterize_bwd_start);
#endif
        auto saved = ctx->get_saved_variables();
        auto means2d = saved[0];
        auto conics = saved[1];
        auto colors = saved[2];
        auto opacities = saved[3];
        // auto backgrounds = saved[4];
        // auto masks = saved[5];
        auto isect_offsets = saved[4];
        auto flatten_ids = saved[5];
        auto render_alphas = saved[6];
        auto last_ids = saved[7];

        int width = ctx->saved_data["width"].toInt();
        int height = ctx->saved_data["height"].toInt();
        int tile_size = ctx->saved_data["tile_size"].toInt();
        bool absgrad = ctx->saved_data["absgrad"].toBool();
        at::optional<torch::Tensor> backgrounds, masks;

        auto v_render_colors = grad_outputs[0];
        auto v_render_alphas = grad_outputs[1];

        auto result = gsplat::rasterize_to_pixels_bwd_tensor(
            means2d, conics, colors, opacities, backgrounds, masks,
            width, height, tile_size, isect_offsets, flatten_ids,
            render_alphas, last_ids, v_render_colors.contiguous(),
            v_render_alphas.contiguous(), absgrad);

        auto v_means2d_abs = std::get<0>(result);
        auto v_means2d = std::get<1>(result);
        auto v_conics = std::get<2>(result);
        auto v_colors = std::get<3>(result);
        auto v_opacities = std::get<4>(result);

        if (absgrad)
        {
            means2d.set_data(means2d.data().abs());
        }

        torch::Tensor v_backgrounds;
        if (ctx->needs_input_grad(4))
        {
            v_backgrounds = (v_render_colors * (1.0 - render_alphas).to(torch::kFloat)).sum({1, 2});
        }
// std::cout << "RasterizeToPixels backward" << std::endl;
// std::cout << v_means2d.mean() << std::endl;
// std::cout << v_conics.mean() << std::endl;
// std::cout << v_colors.mean() << std::endl;
// std::cout << v_opacities.mean() << std::endl;
#ifdef LOGBACKWARDTIME
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &rasterize_bwd_end);
        auto rasterize_bwd_time = getDuration(rasterize_bwd_start, rasterize_bwd_end);
        std::cout << "Rasterize bwd time: " << rasterize_bwd_time << std::endl;
#endif
        return {
            v_means2d, v_conics, v_colors, v_opacities, v_backgrounds,
            torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
            torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};

class RasterizeToPixelsGes : public Function<RasterizeToPixelsGes>
{
public:
    static variable_list forward(
        AutogradContext *ctx,
        torch::Tensor means2d,
        torch::Tensor conics,
        torch::Tensor colors,
        torch::Tensor opacities,
        torch::Tensor ref_depth_map,
        torch::Tensor base_color_map,
        at::optional<torch::Tensor> backgrounds,
        at::optional<torch::Tensor> masks,
        int width,
        int height,
        int tile_size,
        torch::Tensor isect_offsets,
        torch::Tensor flatten_ids,
        bool absgrad,
        float delta_depth)
    {

        if (backgrounds.has_value())
            backgrounds = backgrounds->contiguous();
        if (masks.has_value())
            masks = masks->contiguous();

        auto result = gsplat::rasterize_to_pixels_fwd_ges_tensor(
            means2d.contiguous(),
            conics.contiguous(),
            colors.contiguous(),
            opacities.contiguous(),
            ref_depth_map.contiguous(),
            base_color_map.contiguous(),
            backgrounds,
            masks,
            width, height, tile_size,
            isect_offsets.contiguous(),
            flatten_ids.contiguous(),
            delta_depth);

        auto render_colors = std::get<0>(result);
        auto render_alphas = std::get<1>(result);
        auto last_ids = std::get<2>(result);

        ctx->save_for_backward({means2d, conics, colors, opacities, ref_depth_map, base_color_map, isect_offsets, flatten_ids, render_alphas, last_ids});
        ctx->saved_data["width"] = width;
        ctx->saved_data["height"] = height;
        ctx->saved_data["tile_size"] = tile_size;
        ctx->saved_data["absgrad"] = absgrad;
        ctx->saved_data["delta_depth"] = delta_depth;
        // ctx->saved_data["backgronds"] = backgrounds;
        // ctx->saved_data["masks"] = masks;
        return {render_colors, render_alphas};
    }

    static variable_list backward(
        AutogradContext *ctx,
        variable_list grad_outputs)
    {
#ifdef LOGBACKWARDTIME
        struct timespec rasterize_bwd_start, rasterize_bwd_end;
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &rasterize_bwd_start);
#endif
        auto saved = ctx->get_saved_variables();
        auto means2d = saved[0];
        auto conics = saved[1];
        auto colors = saved[2];
        auto opacities = saved[3];
        auto ref_depth_map = saved[4];
        auto base_color_map = saved[5];

        // auto backgrounds = saved[4];
        // auto masks = saved[5];
        auto isect_offsets = saved[6];
        auto flatten_ids = saved[7];
        auto render_alphas = saved[8];
        auto last_ids = saved[9];

        int width = ctx->saved_data["width"].toInt();
        int height = ctx->saved_data["height"].toInt();
        int tile_size = ctx->saved_data["tile_size"].toInt();
        float delta_depth = ctx->saved_data["delta_depth"].toDouble();
        bool absgrad = ctx->saved_data["absgrad"].toBool();
        at::optional<torch::Tensor> backgrounds, masks;

        auto v_render_colors = grad_outputs[0];
        auto v_render_alphas = grad_outputs[1];
        auto result = gsplat::rasterize_to_pixels_bwd_ges_tensor(
            means2d, conics, colors, opacities, ref_depth_map, base_color_map, backgrounds, masks, width, height, tile_size, isect_offsets, flatten_ids, delta_depth, render_alphas, last_ids, v_render_colors.contiguous(),
            v_render_alphas.contiguous(), absgrad);

        auto v_means2d_abs = std::get<0>(result);
        auto v_means2d = std::get<1>(result);
        auto v_conics = std::get<2>(result);
        auto v_colors = std::get<3>(result);
        auto v_opacities = std::get<4>(result);

        if (absgrad)
        {
            means2d.set_data(means2d.data().abs());
        }

        torch::Tensor v_backgrounds;
        if (ctx->needs_input_grad(4))
        {
            v_backgrounds = (v_render_colors * (1.0 - render_alphas).to(torch::kFloat)).sum({1, 2});
        }
// std::cout << "RasterizeToPixels backward" << std::endl;
// std::cout << v_means2d[0].mean() << std::endl;
// std::cout << v_conics[0].mean() << std::endl;
// std::cout << v_colors[0].mean() << std::endl;
// std::cout << v_opacities.mean() << std::endl;
// std::cout << v_opacities.masked_select(v_opacities != 0).sizes()[0] << std::endl;
#ifdef LOGBACKWARDTIME
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &rasterize_bwd_end);
        auto rasterize_bwd_time = getDuration(rasterize_bwd_start, rasterize_bwd_end);
        std::cout << "Rasterize bwd time: " << rasterize_bwd_time << std::endl;
#endif
        return {
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
            torch::Tensor(),
            torch::Tensor(),
            v_backgrounds,
            torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
            torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};

class RasterizeToPixelsGes_NewParallel : public Function<RasterizeToPixelsGes_NewParallel>
{
public:
    static variable_list forward(
        AutogradContext *ctx,
        torch::Tensor means2d,
        torch::Tensor conics,
        torch::Tensor colors,
        torch::Tensor opacities,
        torch::Tensor radiis,
        torch::Tensor ref_depth_map,
        torch::Tensor base_color_map,
        at::optional<torch::Tensor> backgrounds,
        at::optional<torch::Tensor> masks,
        int width,
        int height,
        int tile_size,
        torch::Tensor isect_offsets,
        torch::Tensor flatten_ids,
        torch::Tensor group_gs_ids,
        torch::Tensor group_starts,
        bool absgrad,
        float delta_depth)
    {

        if (backgrounds.has_value())
            backgrounds = backgrounds->contiguous();
        if (masks.has_value())
            masks = masks->contiguous();

        auto result = gsplat::rasterize_to_pixels_fwd_ges_tensor(
            means2d.contiguous(),
            conics.contiguous(),
            colors.contiguous(),
            opacities.contiguous(),
            ref_depth_map.contiguous(),
            base_color_map.contiguous(),
            backgrounds,
            masks,
            width, height, tile_size,
            isect_offsets.contiguous(),
            flatten_ids.contiguous(),
            delta_depth);

        auto render_colors = std::get<0>(result);
        auto render_alphas = std::get<1>(result);
        auto last_ids = std::get<2>(result);

        ctx->save_for_backward({means2d, conics, colors, opacities, radiis, ref_depth_map, base_color_map, render_alphas, group_gs_ids, group_starts});
        ctx->saved_data["width"] = width;
        ctx->saved_data["height"] = height;
        ctx->saved_data["tile_size"] = tile_size;
        ctx->saved_data["absgrad"] = absgrad;
        ctx->saved_data["delta_depth"] = delta_depth;
        ctx->saved_data["n_isects"] = flatten_ids.size(0);

        return {render_colors, render_alphas};
    }

    static variable_list backward(
        AutogradContext *ctx,
        variable_list grad_outputs)
    {
#ifdef LOGBACKWARDTIME
        struct timespec rasterize_bwd_start, rasterize_bwd_end;
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &rasterize_bwd_start);
#endif
        auto saved = ctx->get_saved_variables();
        auto means2d = saved[0];
        auto conics = saved[1];
        auto colors = saved[2];
        auto opacities = saved[3];
        auto radiis = saved[4];
        auto ref_depth_map = saved[5];
        auto base_color_map = saved[6];
        auto render_alphas = saved[7];
        auto group_gs_ids = saved[8];
        auto group_starts = saved[9];

        int width = ctx->saved_data["width"].toInt();
        int height = ctx->saved_data["height"].toInt();
        int n_isects = ctx->saved_data["n_isects"].toInt();
        float delta_depth = ctx->saved_data["delta_depth"].toDouble();
        bool absgrad = ctx->saved_data["absgrad"].toBool();
        at::optional<torch::Tensor> backgrounds;

        auto v_render_colors = grad_outputs[0];
        auto v_render_alphas = grad_outputs[1];
        auto result = gsplat::rasterize_to_pixels_bwd_ges_gs_parallel_tensor(
            means2d, conics, colors, opacities, radiis, ref_depth_map, base_color_map, backgrounds, width, height, n_isects, group_gs_ids, group_starts, delta_depth, render_alphas, v_render_colors.contiguous(),
            v_render_alphas.contiguous(), absgrad);

        auto v_means2d_abs = std::get<0>(result);
        auto v_means2d = std::get<1>(result);
        auto v_conics = std::get<2>(result);
        auto v_colors = std::get<3>(result);
        auto v_opacities = std::get<4>(result);

        if (absgrad)
        {
            means2d.set_data(means2d.data().abs());
        }

// torch::Tensor v_backgrounds;
// if (ctx->needs_input_grad(4))
// {
//     v_backgrounds = (v_render_colors * (1.0 - render_alphas).to(torch::kFloat)).sum({1, 2});
// }
// std::cout << "RasterizeToPixels new parallel backward" << std::endl;
// std::cout << v_means2d[0].mean() << std::endl;
// std::cout << v_conics[0].mean() << std::endl;
// std::cout << v_colors[0].mean() << std::endl;
// std::cout << v_opacities.mean() << std::endl;
#ifdef LOGBACKWARDTIME
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &rasterize_bwd_end);
        auto rasterize_bwd_time = getDuration(rasterize_bwd_start, rasterize_bwd_end);
        std::cout << "Rasterize bwd time: " << rasterize_bwd_time << std::endl;
#endif
        return {
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};

class FusedSSIMMap : public Function<FusedSSIMMap>
{
public:
    static torch::Tensor forward(AutogradContext *ctx,
                                 float C1,
                                 float C2,
                                 torch::Tensor img1,
                                 torch::Tensor img2,
                                 std::string padding,
                                 bool train)
    {
        auto result = fusedssim(C1, C2, img1, img2, train);
        auto ssim_map = std::get<0>(result);
        auto dm_dmu1 = std::get<1>(result);
        auto dm_dsigma1_sq = std::get<2>(result);
        auto dm_dsigma12 = std::get<3>(result);

        if (padding == "valid")
        {
            ssim_map = ssim_map.slice(2, 5, -5).slice(3, 5, -5);
        }

        ctx->save_for_backward({img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12});
        ctx->saved_data["C1"] = C1;
        ctx->saved_data["C2"] = C2;
        ctx->saved_data["padding"] = padding;

        return ssim_map;
    }

    static torch::autograd::tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs)
    {
        auto saved = ctx->get_saved_variables();
        auto img1 = saved[0];
        auto img2 = saved[1];
        auto dm_dmu1 = saved[2];
        auto dm_dsigma1_sq = saved[3];
        auto dm_dsigma12 = saved[4];

        double C1 = ctx->saved_data["C1"].toDouble();
        double C2 = ctx->saved_data["C2"].toDouble();
        std::string padding = ctx->saved_data["padding"].toStringRef();

        auto dL_dmap = grad_outputs[0];
        if (padding == "valid")
        {
            auto new_dL_dmap = torch::zeros_like(img1);
            new_dL_dmap.slice(2, 5, -5).slice(3, 5, -5) = dL_dmap;
            dL_dmap = new_dL_dmap;
        }

        auto grad = fusedssim_backward(C1, C2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12);

        return {torch::Tensor(), torch::Tensor(), grad, torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};

variable_list
isectTiles(torch::Tensor means2d,
           torch::Tensor radii,
           torch::Tensor depths,
           int tile_size,
           int tile_width,
           int tile_height,
           bool sort = true);

torch::Tensor isectOffsetEncode(torch::Tensor isect_ids, int n_cameras, int tile_width, int tile_height);

torch::Tensor simpleKNN(torch::Tensor points);

variable_list
isectTilesNoDepth(torch::Tensor means2d,
                  torch::Tensor radii,
                  torch::Tensor depths,
                  int tile_size,
                  int tile_width,
                  int tile_height,
                  bool sort = true);

torch::Tensor isectOffsetEncodeNoDepth(torch::Tensor isect_ids, int n_cameras, int tile_width, int tile_height);

int degFromSh(int numBases);

torch::Tensor rgb2sh(const torch::Tensor &rgb);

torch::Tensor sh2rgb(const torch::Tensor &sh);

int numShBases(int degree);

#endif
