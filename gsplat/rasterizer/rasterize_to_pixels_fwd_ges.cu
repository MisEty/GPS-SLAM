#include "bindings.h"
#include "helpers.cuh"
#include "types.cuh"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace gsplat
{

    namespace cg = cooperative_groups;

    /****************************************************************************
     * Rasterization to Pixels Forward Pass
     ****************************************************************************/

    template <uint32_t COLOR_DIM, typename S>
    __global__ void rasterize_to_pixels_fwd_ges_kernel(
        const uint32_t C,
        const uint32_t N,
        const uint32_t n_isects,
        const bool packed,
        const float delta_depth,
        const vec2<S> *__restrict__ means2d,      // [C, N, 2] or [nnz, 2]
        const vec3<S> *__restrict__ conics,       // [C, N, 3] or [nnz, 3]
        const S *__restrict__ colors,             // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
        const S *__restrict__ opacities,          // [C, N] or [nnz]
        const float *__restrict__ ref_depth_map,  // [C, H, W]
        const float *__restrict__ base_color_map, // [C, H, W]
        const S *__restrict__ backgrounds,        // [C, COLOR_DIM]
        const bool *__restrict__ masks,           // [C, tile_height, tile_width]
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        const uint32_t tile_width,
        const uint32_t tile_height,
        const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
        const int32_t *__restrict__ flatten_ids,  // [n_isects]
        S *__restrict__ render_colors,            // [C, image_height, image_width, COLOR_DIM]
        S *__restrict__ render_alphas,            // [C, image_height, image_width, 1]
        int32_t *__restrict__ last_ids            // [C, image_height, image_width]
    )
    {
        // each thread draws one pixel, but also timeshares caching gaussians in a
        // shared tile

        auto block = cg::this_thread_block();
        int32_t camera_id = block.group_index().x;
        int32_t tile_id =
            block.group_index().y * tile_width + block.group_index().z;
        uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
        uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

        tile_offsets += camera_id * tile_height * tile_width;
        render_colors += camera_id * image_height * image_width * COLOR_DIM;
        render_alphas += camera_id * image_height * image_width;
        last_ids += camera_id * image_height * image_width;
        if (backgrounds != nullptr)
        {
            backgrounds += camera_id * COLOR_DIM;
        }
        if (masks != nullptr)
        {
            masks += camera_id * tile_height * tile_width;
        }

        S px = (S)j + 0.5f;
        S py = (S)i + 0.5f;
        int32_t pix_id = i * image_width + j;

        // return if out of bounds
        // keep not rasterizing threads around for reading data
        bool inside = (i < image_height && j < image_width);
        bool done = !inside;

        // when the mask is provided, render the background color and return
        // if this tile is labeled as False
        if (masks != nullptr && inside && !masks[tile_id])
        {
            for (uint32_t k = 0; k < COLOR_DIM; ++k)
            {
                render_colors[pix_id * COLOR_DIM + k] =
                    backgrounds == nullptr ? 0.0f : backgrounds[k];
            }
            return;
        }

        // have all threads in tile process the same gaussians in batches
        // first collect gaussians between range.x and range.y in batches
        // which gaussians to look through in this tile
        int32_t range_start = tile_offsets[tile_id];
        int32_t range_end =
            (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
                ? n_isects
                : tile_offsets[tile_id + 1];
        const uint32_t block_size = block.size();
        uint32_t num_batches =
            (range_end - range_start + block_size - 1) / block_size;
        // if (block.thread_index().y == 0 && block.thread_index().x == 0)
        //     printf("tile id: %d, %d, num_batches: %d\n", block.group_index().y, block.group_index().x, num_batches);
        extern __shared__ int s[];
        int32_t *id_batch = (int32_t *)s; // [block_size]
        vec3<S> *xy_opacity_batch =
            reinterpret_cast<vec3<float> *>(&id_batch[block_size]); // [block_size]
        vec3<S> *conic_batch =
            reinterpret_cast<vec3<float> *>(&xy_opacity_batch[block_size]); // [block_size]

        // // current visibility left to render
        // // transmittance is gonna be used in the backward pass which requires a high
        // // numerical precision so we use double for it. However double make bwd 1.5x
        // // slower so we stick with float for now.
        // S T = 1.0f;
        // index of most recent gaussian to write to this thread's pixel
        uint32_t cur_idx = 0;

        // collect and process batches of gaussians
        // each thread loads one gaussian at a time before rasterizing its
        // designated pixel
        uint32_t tr = block.thread_rank();

        // init depth cut parameters
        float weight_sum = 0;
        float ref_depth = inside ? ref_depth_map[pix_id] : 0;
        // int check_x = 1151;
        // int check_y = 654;
        // int count = 0;
        // if (i == 516 && j == 768)
        // {
        //     printf("ref_depth: %f\n", ref_depth);
        // }
        S pix_out[COLOR_DIM] = {0.f};
        for (uint32_t b = 0; b < num_batches; ++b)
        {
            // resync all threads before beginning next batch
            // end early if entire tile is done
            if (__syncthreads_count(done) >= block_size)
            {
                break;
            }

            // each thread fetch 1 gaussian from front to back
            // index of gaussian to load
            uint32_t batch_start = range_start + block_size * b;
            uint32_t idx = batch_start + tr;
            if (idx < range_end)
            {
                int32_t g = flatten_ids[idx]; // flatten index in [C * N] or [nnz]
                id_batch[tr] = g;
                const vec2<S> xy = means2d[g];
                const S opac = opacities[g];
                xy_opacity_batch[tr] = {xy.x, xy.y, opac};
                conic_batch[tr] = conics[g];
            }

            // wait for other threads to collect the gaussians in batch
            block.sync();

            // process gaussians in the current batch for this pixel
            uint32_t batch_size = min(block_size, range_end - batch_start);
            for (uint32_t t = 0; (t < batch_size) && !done; ++t)
            {
                // if (collected_c_depths[j] > pix_depth)
                //     continue;
                int32_t g = id_batch[t];
                const S *c_ptr = colors + g * COLOR_DIM;
                if (c_ptr[3] > ref_depth + delta_depth)
                    continue;
                const vec3<S> conic = conic_batch[t];
                const vec3<S> xy_opac = xy_opacity_batch[t];
                const S opac = xy_opac.z;
                const vec2<S> delta = {xy_opac.x - px, xy_opac.y - py};
                const S sigma = 0.5f * (conic.x * delta.x * delta.x +
                                        conic.z * delta.y * delta.y) +
                                conic.y * delta.x * delta.y;
                S alpha = min(0.999f, opac * __expf(-sigma));
                // if (i == check_y && j == check_x)
                // {
                //     printf("COLOR DIM: %d, color: %f, %f, %f, depth: %f, alpha: %f\n", COLOR_DIM, c_ptr[0], c_ptr[1], c_ptr[2], c_ptr[3], alpha);
                // }
                if (sigma < 0.f || alpha < 1.f / 255.f)
                {
                    continue;
                }

                // if (i == 516 && j == 768)
                // {
                //     printf("gid: %d, color: %f, %f, %f, depth: %f, alpha: %f\n", g, c_ptr[0], c_ptr[1], c_ptr[2], c_ptr[3], alpha);
                // }
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k)
                {
                    pix_out[k] += c_ptr[k] * alpha;
                }
                weight_sum += alpha;
                cur_idx = batch_start + t;
                // count++;
            }
        }

        if (inside)
        {
            // Here T is the transmittance AFTER the last gaussian in this pixel.
            // We (should) store double precision as T would be used in backward
            // pass and it can be very small and causing large diff in gradients
            // with float32. However, double precision makes the backward pass 1.5x
            // slower so we stick with float for now.
            render_alphas[pix_id] = weight_sum; // now color is weight_sum
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t k = 0; k < COLOR_DIM; ++k)
            {
                render_colors[pix_id * COLOR_DIM + k] = pix_out[k];
            }
            // index in bin of last gaussian in this pixel
            last_ids[pix_id] = static_cast<int32_t>(cur_idx);
        }
        // if (i == check_y && j == check_x)
        // {
        //     // printf("res: color: %f, %f, %f, depth: %f, weight: %f\n", pix_out[0], pix_out[1], pix_out[2], pix_out[3], weight_sum);
        //     printf("acc gs count: %d\n", count);
        // }
    }

    template <uint32_t CDIM>
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> call_kernel_with_dim(
        // Gaussian parameters
        const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
        const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
        const torch::Tensor &colors,                    // [C, N, channels] or [nnz, channels]
        const torch::Tensor &opacities,                 // [C, N]  or [nnz]
        const torch::Tensor &ref_depth_map,             // [C, H, W]
        const torch::Tensor &base_color_map,            // [C, H, W]
        const at::optional<torch::Tensor> &backgrounds, // [C, channels]
        const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // intersections
        const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
        const torch::Tensor &flatten_ids,  // [n_isects]
        // depth cut range
        const float delta_depth)
    {
        GSPLAT_DEVICE_GUARD(means2d);
        GSPLAT_CHECK_INPUT(means2d);
        GSPLAT_CHECK_INPUT(conics);
        GSPLAT_CHECK_INPUT(colors);
        GSPLAT_CHECK_INPUT(opacities);
        GSPLAT_CHECK_INPUT(ref_depth_map);
        GSPLAT_CHECK_INPUT(base_color_map);
        GSPLAT_CHECK_INPUT(tile_offsets);
        GSPLAT_CHECK_INPUT(flatten_ids);
        if (backgrounds.has_value())
        {
            GSPLAT_CHECK_INPUT(backgrounds.value());
        }
        if (masks.has_value())
        {
            GSPLAT_CHECK_INPUT(masks.value());
        }
        bool packed = means2d.dim() == 2;

        uint32_t C = tile_offsets.size(0);         // number of cameras
        uint32_t N = packed ? 0 : means2d.size(1); // number of gaussians
        uint32_t channels = colors.size(-1);
        uint32_t tile_height = tile_offsets.size(1);
        uint32_t tile_width = tile_offsets.size(2);
        uint32_t n_isects = flatten_ids.size(0);

        // Each block covers a tile on the image. In total there are
        // C * tile_height * tile_width blocks.
        dim3 threads = {tile_size, tile_size, 1};
        dim3 blocks = {C, tile_height, tile_width};

        torch::Tensor renders = torch::empty(
            {C, image_height, image_width, channels},
            means2d.options().dtype(torch::kFloat32));
        torch::Tensor alphas = torch::empty(
            {C, image_height, image_width, 1},
            means2d.options().dtype(torch::kFloat32));
        torch::Tensor last_ids = torch::empty(
            {C, image_height, image_width}, means2d.options().dtype(torch::kInt32));

        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        const uint32_t shared_mem =
            tile_size * tile_size *
            (sizeof(int32_t) + sizeof(vec3<float>) + sizeof(vec3<float>));

        // TODO: an optimization can be done by passing the actual number of
        // channels into the kernel functions and avoid necessary global memory
        // writes. This requires moving the channel padding from python to C side.
        if (cudaFuncSetAttribute(
                rasterize_to_pixels_fwd_ges_kernel<CDIM, float>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                shared_mem) != cudaSuccess)
        {
            AT_ERROR(
                "Failed to set maximum shared memory size (requested ",
                shared_mem,
                " bytes), try lowering tile_size.");
        }
        // struct timespec debug_start, debug_end;
        // cudaDeviceSynchronize();
        // clock_gettime(CLOCK_MONOTONIC, &debug_start);
        rasterize_to_pixels_fwd_ges_kernel<CDIM, float>
            <<<blocks, threads, shared_mem, stream>>>(
                C,
                N,
                n_isects,
                packed,
                delta_depth,
                reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
                reinterpret_cast<vec3<float> *>(conics.data_ptr<float>()),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                ref_depth_map.data_ptr<float>(),
                base_color_map.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
                image_width,
                image_height,
                tile_size,
                tile_width,
                tile_height,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                renders.data_ptr<float>(),
                alphas.data_ptr<float>(),
                last_ids.data_ptr<int32_t>());
        // cudaDeviceSynchronize();
        // clock_gettime(CLOCK_MONOTONIC, &debug_end);
        // auto debug_time = double(debug_end.tv_nsec - debug_start.tv_nsec) / 1000000;
        // printf("rasterize debug time: %f\n", debug_time);
        return std::make_tuple(renders, alphas, last_ids);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    rasterize_to_pixels_fwd_ges_tensor(
        // Gaussian parameters
        const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
        const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
        const torch::Tensor &colors,                    // [C, N, channels] or [nnz, channels]
        const torch::Tensor &opacities,                 // [C, N]  or [nnz]
        const torch::Tensor &ref_depth_map,             // [C, H, W]
        const torch::Tensor &base_color_map,            // [C, H, W]
        const at::optional<torch::Tensor> &backgrounds, // [C, channels]
        const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // intersections
        const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
        const torch::Tensor &flatten_ids,  // [n_isects]
        // depth cut range
        const float delta_depth)
    {
        GSPLAT_CHECK_INPUT(colors);
        uint32_t channels = colors.size(-1);

#define __GS__CALL_(N)                  \
    case N:                             \
        return call_kernel_with_dim<N>( \
            means2d,                    \
            conics,                     \
            colors,                     \
            opacities,                  \
            ref_depth_map,              \
            base_color_map,             \
            backgrounds,                \
            masks,                      \
            image_width,                \
            image_height,               \
            tile_size,                  \
            tile_offsets,               \
            flatten_ids,                \
            delta_depth);

        // TODO: an optimization can be done by passing the actual number of
        // channels into the kernel functions and avoid necessary global memory
        // writes. This requires moving the channel padding from python to C side.
        switch (channels)
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
            AT_ERROR("Unsupported number of channels: ", channels);
        }
    }

} // namespace gsplat