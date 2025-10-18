#include "bindings.h"
#include "helpers.cuh"
#include "types.cuh"
#include <fstream>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace gsplat
{

    namespace cg = cooperative_groups;

    /****************************************************************************
     * Rasterization to Pixels Backward Pass
     ****************************************************************************/
    template <uint32_t COLOR_DIM, typename S>
    __global__ void temp_bwd_kernel(
        const uint32_t n_groups,
        const int32_t *__restrict__ group_gs_ids, // [C, N] or [nnz]
        const int32_t *__restrict__ group_starts, // [C, N] or [nnz]
        // const int32_t *__restrict__ visible_indices,      // [C, N] or [nnz]
        // const int32_t *__restrict__ per_group_visible_id, // [C, N] or [nnz]
        // const int32_t *__restrict__ groups_num_per_gauss,
        // const int32_t *__restrict__ cum_groups_num_per_gauss,
        const vec2<S> *__restrict__ means2d,     // [C, N, 2] or [nnz, 2]
        const vec3<S> *__restrict__ conics,      // [C, N, 3] or [nnz, 3]
        const S *__restrict__ colors,            // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
        const S *__restrict__ opacities,         // [C, N] or [nnz]
        const int32_t *__restrict__ radiis,      // [C, N, 1] or [nnz]
        const float *__restrict__ ref_depth_map, // [C, H, W]
        // depth cut range
        const float delta_depth,
        const uint32_t image_width,
        const uint32_t image_height,
        // grad outputs
        const S *__restrict__ v_render_colors, // [C, image_height, image_width,
                                               // COLOR_DIM]
        const S *__restrict__ v_render_alphas, // [C, image_height, image_width, 1]
        // grad inputs
        vec2<S> *__restrict__ v_means2d, // [C, N, 2] or [nnz, 2]
        vec3<S> *__restrict__ v_conics,  // [C, N, 3] or [nnz, 3]
        S *__restrict__ v_colors,        // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
        S *__restrict__ v_opacities      // [C, N] or [nnz]
    )
    {
        const uint32_t global_id = cg::this_grid().thread_rank();
        int groups_num_per_thread = 32;
        // 按照warp启动，所以实际启动的线程数量必须是32的倍数
        if (global_id >= ((n_groups + 32 - 1) / 32) * 32)
        {
            return;
        }
        auto block = cg::this_thread_block();
        cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

        const uint32_t block_id = block.thread_rank();
        const uint32_t lane_id = warp.thread_rank();
        const uint32_t warp_id = global_id / warpSize;

        // 开始遍历这个线程需要处理的group
        for (int curr_group = 0; curr_group < groups_num_per_thread; curr_group++)
        {
            const uint32_t group_id = warp_id * groups_num_per_thread + curr_group;
            if (group_id >= n_groups)
            {
                return;
            }
            const int32_t gs_id = group_gs_ids[group_id];

            // get target gs parameters
            int32_t radii = radiis[gs_id];
            const vec2<S> xy = means2d[gs_id];
            const S opac = opacities[gs_id];
            const vec3<S> conic = conics[gs_id];
            S rgb[COLOR_DIM];
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t k = 0; k < COLOR_DIM; ++k)
            {
                rgb[k] = colors[gs_id * COLOR_DIM + k];
            }

            const int32_t x_min = int32_t(xy[0]) - radii;
            const int32_t x_max = int32_t(xy[0]) + radii;
            const int32_t y_min = int32_t(xy[1]) - radii;
            const int32_t y_max = int32_t(xy[1]) + radii;

            // get target pixel id
            const uint32_t group_start = group_starts[group_id];

            const uint32_t inside_pixel_id = (group_id - group_start) * groups_num_per_thread + lane_id;
            int32_t j = x_min + 1 + inside_pixel_id % (x_max - x_min);
            int32_t i = y_min + 1 + inside_pixel_id / (x_max - x_min);
            bool valid = (i < image_height && j < image_width && i >= 0 && j >= 0);
            if (i > y_max)
                valid = false;

            S v_render_c[COLOR_DIM];
            S v_render_a;
            S alpha;
            vec2<S> delta;
            S vis;
            S sigma;
            int32_t pix_id;
            S ref_depth;
            if (valid)
            {
                pix_id = min(i * image_width + j, image_width * image_height - 1);
                ref_depth = ref_depth_map[pix_id];
                const S px = (S)j + 0.5f;
                const S py = (S)i + 0.5;
                delta = {xy.x - px, xy.y - py};
                sigma = 0.5f * (conic.x * delta.x * delta.x +
                                conic.z * delta.y * delta.y) +
                        conic.y * delta.x * delta.y;
                vis = __expf(-sigma); //
                alpha = min(0.999f, opac * vis);
                if (sigma < 0.f || alpha < 1.f / 255.f || rgb[3] > ref_depth + delta_depth)
                {
                    valid = false;
                }
                // get df/d_out for this pixel
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k)
                {
                    v_render_c[k] = v_render_colors[pix_id * COLOR_DIM + k];
                }
                v_render_a = v_render_alphas[pix_id];
            }

            // if all threads are inactive in this warp, skip this loop
            if (!warp.any(valid))
            {
                continue;
            }
            S v_rgb_local[COLOR_DIM] = {0.f};
            vec3<S> v_conic_local = {0.f, 0.f, 0.f};
            vec2<S> v_xy_local = {0.f, 0.f};
            S v_opacity_local = 0.f;
            // gpuAtomicAdd(valid_count, 1);
            if (valid)
            {
                // color
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k)
                {
                    v_rgb_local[k] += alpha * v_render_c[k];
                }
                // alpha
                // rgb loss contribution from this pixel
                S v_alpha = 0.f;
                for (uint32_t k = 0; k < COLOR_DIM; ++k)
                {
                    v_alpha += rgb[k] * v_render_c[k];
                }
                // alpha loss
                v_alpha += v_render_a;
                // conic and xy
                if (opac * vis <= 0.999f)
                {
                    const S v_sigma = -opac * vis * v_alpha;
                    v_conic_local = {
                        0.5f * v_sigma * delta.x * delta.x,
                        v_sigma * delta.x * delta.y,
                        0.5f * v_sigma * delta.y * delta.y};

                    v_xy_local = {
                        v_sigma * (conic.x * delta.x + conic.y * delta.y),
                        v_sigma * (conic.y * delta.x + conic.z * delta.y)};
                    v_opacity_local = vis * v_alpha;
                }
            }

            // 计算warp内的梯度和
            warpSum<COLOR_DIM, S>(v_rgb_local, warp);
            warpSum<decltype(warp), S>(v_conic_local, warp);
            warpSum<decltype(warp), S>(v_xy_local, warp);
            warpSum<decltype(warp), S>(v_opacity_local, warp);

            if (warp.thread_rank() == 0)
            {

                S *v_rgb_ptr = (S *)(v_colors) + COLOR_DIM * gs_id;
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k)
                {
                    gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                }

                S *v_conic_ptr = (S *)(v_conics) + 3 * gs_id;
                gpuAtomicAdd(v_conic_ptr, v_conic_local.x);
                gpuAtomicAdd(v_conic_ptr + 1, v_conic_local.y);
                gpuAtomicAdd(v_conic_ptr + 2, v_conic_local.z);

                S *v_xy_ptr = (S *)(v_means2d) + 2 * gs_id;
                gpuAtomicAdd(v_xy_ptr, v_xy_local.x);
                gpuAtomicAdd(v_xy_ptr + 1, v_xy_local.y);
                gpuAtomicAdd(v_opacities + gs_id, v_opacity_local);
            }
        }
    }

    template <uint32_t CDIM>
    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    call_kernel_with_dim(
        // Gaussian parameters
        const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
        const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
        const torch::Tensor &colors,                    // [C, N, 3] or [nnz, 3]
        const torch::Tensor &opacities,                 // [C, N] or [nnz]
        const torch::Tensor &radiis,                    // [C, N, 1] or [nnz]
        const torch::Tensor &ref_depth_map,             // [C, H, W]
        const torch::Tensor &base_color_map,            // [C, H, W]
        const at::optional<torch::Tensor> &backgrounds, // [C, 3]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t n_isects,
        // intersects
        const torch::Tensor &group_gs_ids,
        const torch::Tensor &group_starts,
        // depth cut range
        const float delta_depth,
        // forward outputs
        const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
        // gradients of outputs
        const torch::Tensor &v_render_colors, // [C, image_height, image_width, 3]
        const torch::Tensor &v_render_alphas, // [C, image_height, image_width, 1]
        // options
        bool absgrad)
    {
        // struct timespec rasterize_bwd_pass0, rasterize_bwd_pass1;
        GSPLAT_DEVICE_GUARD(means2d);
        GSPLAT_CHECK_INPUT(means2d);
        GSPLAT_CHECK_INPUT(conics);
        GSPLAT_CHECK_INPUT(colors);
        GSPLAT_CHECK_INPUT(opacities);
        GSPLAT_CHECK_INPUT(radiis);

        GSPLAT_CHECK_INPUT(ref_depth_map);
        GSPLAT_CHECK_INPUT(base_color_map);
        GSPLAT_CHECK_INPUT(render_alphas);
        GSPLAT_CHECK_INPUT(v_render_colors);
        GSPLAT_CHECK_INPUT(v_render_alphas);
        if (backgrounds.has_value())
        {
            GSPLAT_CHECK_INPUT(backgrounds.value());
        }

        bool packed = means2d.dim() == 2;
        uint32_t COLOR_DIM = colors.size(-1);
        // Each block covers a tile on the image. In total there are
        // C * tile_height * tile_width blocks.
        torch::Tensor v_means2d = torch::zeros_like(means2d);
        torch::Tensor v_conics = torch::zeros_like(conics);
        torch::Tensor v_colors = torch::zeros_like(colors);
        torch::Tensor v_opacities = torch::zeros_like(opacities);
        torch::Tensor v_means2d_abs;
        if (n_isects)
        {
            int64_t n_groups = group_gs_ids.size(0);
            // cudaDeviceSynchronize();
            // clock_gettime(CLOCK_MONOTONIC, &rasterize_bwd_pass0);

            temp_bwd_kernel<CDIM, float>
                <<<(n_groups + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS, GSPLAT_N_THREADS>>>(
                    n_groups,
                    group_gs_ids.data_ptr<int32_t>(),
                    group_starts.data_ptr<int32_t>(),
                    reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
                    reinterpret_cast<vec3<float> *>(conics.data_ptr<float>()),
                    colors.data_ptr<float>(),
                    opacities.data_ptr<float>(),
                    radiis.data_ptr<int32_t>(),
                    ref_depth_map.data_ptr<float>(),
                    delta_depth,
                    image_width,
                    image_height,
                    v_render_colors.data_ptr<float>(),
                    v_render_alphas.data_ptr<float>(),
                    reinterpret_cast<vec2<float> *>(v_means2d.data_ptr<float>()),
                    reinterpret_cast<vec3<float> *>(v_conics.data_ptr<float>()),
                    v_colors.data_ptr<float>(),
                    v_opacities.data_ptr<float>());
            // cudaDeviceSynchronize();
            // clock_gettime(CLOCK_MONOTONIC, &rasterize_bwd_pass1);

            // auto rasterize_bwd_pass1_time = float(rasterize_bwd_pass1.tv_nsec - rasterize_bwd_pass0.tv_nsec) / 1000000;
            // // std::cout << "valid opac grad: " << v_opacities.masked_select(v_opacities != 0).sizes()[0] << std::endl;
            // // std::cout << "compute count: " << compute_count.item() << std::endl;
            // // std::cout << "valid count: " << valid_count.item() << std::endl;
            // std::cout << "Rasterize bwd pass1 time: " << rasterize_bwd_pass1_time << std::endl;
        }

        return std::make_tuple(
            v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities);
    }

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    rasterize_to_pixels_bwd_ges_gs_parallel_tensor(
        // Gaussian parameters
        const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
        const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
        const torch::Tensor &colors,                    // [C, N, 3] or [nnz, 3]
        const torch::Tensor &opacities,                 // [C, N] or [nnz]
        const torch::Tensor &radiis,                    // [C, N, 1] or [nnz]
        const torch::Tensor &ref_depth_map,             // [C, H, W]
        const torch::Tensor &base_color_map,            // [C, H, W]
        const at::optional<torch::Tensor> &backgrounds, // [C, 3]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t n_isects,
        // intersects
        const torch::Tensor &group_gs_ids,
        const torch::Tensor &group_starts,
        // depth cut range
        const float delta_depth,
        // forward outputs
        const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
        // gradients of outputs
        const torch::Tensor &v_render_colors, // [C, image_height, image_width, 3]
        const torch::Tensor &v_render_alphas, // [C, image_height, image_width, 1]
        // options
        bool absgrad)
    {
        GSPLAT_CHECK_INPUT(colors);
        uint32_t COLOR_DIM = colors.size(-1);
#define __GS__CALL_(N)                  \
    case N:                             \
        return call_kernel_with_dim<N>( \
            means2d,                    \
            conics,                     \
            colors,                     \
            opacities,                  \
            radiis,                     \
            ref_depth_map,              \
            base_color_map,             \
            backgrounds,                \
            image_width,                \
            image_height,               \
            n_isects,                   \
            group_gs_ids,               \
            group_starts,               \
            delta_depth,                \
            render_alphas,              \
            v_render_colors,            \
            v_render_alphas,            \
            absgrad);

        switch (COLOR_DIM)
        {
            __GS__CALL_(1)
            __GS__CALL_(2)
            __GS__CALL_(3)
            __GS__CALL_(4)
            __GS__CALL_(5)
            __GS__CALL_(8)
            __GS__CALL_(9)
            __GS__CALL_(16)
            __GS__CALL_(17)
            __GS__CALL_(32)
            __GS__CALL_(33)
            __GS__CALL_(64)
            __GS__CALL_(65)
            __GS__CALL_(128)
            __GS__CALL_(129)
            __GS__CALL_(256)
            __GS__CALL_(257)
            __GS__CALL_(512)
            __GS__CALL_(513)
        default:
            AT_ERROR("Unsupported number of channels: ", COLOR_DIM);
        }
    }

} // namespace gsplat