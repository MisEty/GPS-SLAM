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
     * Gaussian Tile Intersection
     ****************************************************************************/

    template <typename T>
    __global__ void isect_tiles_no_depth(
        // if the data is [C, N, ...] or [nnz, ...] (packed)
        const bool packed,
        // parallelize over C * N, only used if packed is False
        const uint32_t C,
        const uint32_t N,
        // parallelize over nnz, only used if packed is True
        const uint32_t nnz,
        const int64_t *__restrict__ camera_ids,   // [nnz] optional
        const int64_t *__restrict__ gaussian_ids, // [nnz] optional
        // data
        const T *__restrict__ means2d,                    // [C, N, 2] or [nnz, 2]
        const int32_t *__restrict__ radii,                // [C, N] or [nnz]
        const T *__restrict__ depths,                     // [C, N] or [nnz]
        const int64_t *__restrict__ cum_tiles_per_gauss,  // [C, N] or [nnz]
        const int64_t *__restrict__ cum_groups_per_gauss, // [C, N] or [nnz]
        const uint32_t tile_size,
        const uint32_t tile_width,
        const uint32_t tile_height,
        const uint32_t tile_n_bits,
        int32_t *__restrict__ tiles_per_gauss,  // [C, N] or [nnz]
        int32_t *__restrict__ groups_per_gauss, // [C, N] or [nnz]
        int64_t *__restrict__ isect_ids,        // [n_isects]
        int32_t *__restrict__ group_gs_ids,     // [n_group]
        int32_t *__restrict__ group_starts,     // [n_group]
        int32_t *__restrict__ flatten_ids       // [n_isects]
    )
    {
        // For now we'll upcast float16 and bfloat16 to float32
        using OpT = typename OpType<T>::type;

        // parallelize over C * N.
        uint32_t idx = cg::this_grid().thread_rank();
        bool first_pass = cum_tiles_per_gauss == nullptr;
        if (idx >= (packed ? nnz : C * N))
        {
            return;
        }

        const OpT radius = radii[idx];
        if (radius <= 0)
        {
            if (first_pass)
            {
                tiles_per_gauss[idx] = 0;
                groups_per_gauss[idx] = 0;
            }
            return;
        }

        vec2<OpT> mean2d = glm::make_vec2(means2d + 2 * idx);

        OpT tile_radius = radius / static_cast<OpT>(tile_size);
        OpT tile_x = mean2d.x / static_cast<OpT>(tile_size);
        OpT tile_y = mean2d.y / static_cast<OpT>(tile_size);

        // tile_min is inclusive, tile_max is exclusive
        uint2 tile_min, tile_max;
        tile_min.x = min(max(0, (uint32_t)floor(tile_x - tile_radius)), tile_width);
        tile_min.y =
            min(max(0, (uint32_t)floor(tile_y - tile_radius)), tile_height);
        tile_max.x = min(max(0, (uint32_t)ceil(tile_x + tile_radius)), tile_width);
        tile_max.y = min(max(0, (uint32_t)ceil(tile_y + tile_radius)), tile_height);

        if (first_pass)
        {
            // first pass only writes out tiles_per_gauss
            tiles_per_gauss[idx] = static_cast<int32_t>(
                (tile_max.y - tile_min.y) * (tile_max.x - tile_min.x));
            groups_per_gauss[idx] = static_cast<int32_t>((4 * radius * radius + 32 - 1) / 32);
            return;
        }

        int64_t cid; // camera id
        if (packed)
        {
            // parallelize over nnz
            cid = camera_ids[idx];
            // gid = gaussian_ids[idx];
        }
        else
        {
            // parallelize over C * N
            cid = idx / N;
            // gid = idx % N;
        }
        // const int64_t cid_enc = cid << (32 + tile_n_bits);
        const int64_t cid_enc = cid << tile_n_bits;
        int64_t depth_id_enc = (int64_t)*(int32_t *)&(depths[idx]);
        int64_t cur_idx = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];
        for (int32_t i = tile_min.y; i < tile_max.y; ++i)
        {
            for (int32_t j = tile_min.x; j < tile_max.x; ++j)
            {
                int64_t tile_id = i * tile_width + j;
                // e.g. tile_n_bits = 22:
                // camera id (10 bits) | tile id (22 bits) | depth (32 bits)
                // isect_ids[cur_idx] = cid_enc | (tile_id << 32) | depth_id_enc;
                isect_ids[cur_idx] = cid_enc | tile_id;

                // the flatten index in [C * N] or [nnz]
                flatten_ids[cur_idx] = static_cast<int32_t>(idx);
                ++cur_idx;
            }
        }
        const int32_t group_num = groups_per_gauss[idx];
        const int32_t curr_group_start = cum_groups_per_gauss[idx];
        for (int32_t i = 0; i < group_num; i++)
        {
            group_gs_ids[curr_group_start + i] = static_cast<int32_t>(idx);
            group_starts[curr_group_start + i] = static_cast<int32_t>(curr_group_start);
        }
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> isect_tiles_tensor_no_depth(
        const torch::Tensor &means2d,                    // [C, N, 2] or [nnz, 2]
        const torch::Tensor &radii,                      // [C, N] or [nnz]
        const torch::Tensor &depths,                     // [C, N] or [nnz]
        const at::optional<torch::Tensor> &camera_ids,   // [nnz]
        const at::optional<torch::Tensor> &gaussian_ids, // [nnz]
        const uint32_t C,
        const uint32_t tile_size,
        const uint32_t tile_width,
        const uint32_t tile_height,
        const bool sort,
        const bool double_buffer)
    {
        // struct timespec first_pass_start, first_pass_end, second_pass_start, second_pass_end, sort_pass_start, sort_pass_end;

        GSPLAT_DEVICE_GUARD(means2d);
        GSPLAT_CHECK_INPUT(means2d);
        GSPLAT_CHECK_INPUT(radii);
        GSPLAT_CHECK_INPUT(depths);
        if (camera_ids.has_value())
        {
            GSPLAT_CHECK_INPUT(camera_ids.value());
        }
        if (gaussian_ids.has_value())
        {
            GSPLAT_CHECK_INPUT(gaussian_ids.value());
        }
        bool packed = means2d.dim() == 2;

        uint32_t N = 0, nnz = 0, total_elems = 0;
        int64_t *camera_ids_ptr = nullptr;
        int64_t *gaussian_ids_ptr = nullptr;
        if (packed)
        {
            nnz = means2d.size(0);
            total_elems = nnz;
            TORCH_CHECK(
                camera_ids.has_value() && gaussian_ids.has_value(),
                "When packed is set, camera_ids and gaussian_ids must be provided.");
            camera_ids_ptr = camera_ids.value().data_ptr<int64_t>();
            gaussian_ids_ptr = gaussian_ids.value().data_ptr<int64_t>();
        }
        else
        {
            N = means2d.size(1); // number of gaussians
            total_elems = C * N;
        }

        uint32_t n_tiles = tile_width * tile_height;
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

        // the number of bits needed to encode the camera id and tile id
        uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
        uint32_t cam_n_bits = (uint32_t)floor(log2(C)) + 1;

        // first pass: compute number of tiles per gaussian
        torch::Tensor tiles_per_gauss =
            torch::empty_like(depths, depths.options().dtype(torch::kInt32));
        torch::Tensor groups_per_gauss =
            torch::zeros({N}, depths.options().dtype(torch::kInt32));

        int64_t n_isects, n_groups;
        torch::Tensor cum_tiles_per_gauss, cum_groups_per_gauss;
        // cudaDeviceSynchronize();
        // clock_gettime(CLOCK_MONOTONIC, &first_pass_start);
        if (total_elems)
        {
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half,
                at::ScalarType::BFloat16,
                means2d.scalar_type(),
                "isect_tiles_total_elems",
                [&]()
                {
                    isect_tiles_no_depth<<<
                        (total_elems + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                        GSPLAT_N_THREADS,
                        0,
                        stream>>>(
                        packed,
                        C,
                        N,
                        nnz,
                        camera_ids_ptr,
                        gaussian_ids_ptr,
                        reinterpret_cast<scalar_t *>(means2d.data_ptr<scalar_t>()),
                        radii.data_ptr<int32_t>(),
                        depths.data_ptr<scalar_t>(),
                        nullptr,
                        nullptr,
                        tile_size,
                        tile_width,
                        tile_height,
                        tile_n_bits,
                        tiles_per_gauss.data_ptr<int32_t>(),
                        groups_per_gauss.data_ptr<int32_t>(),
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr);
                });
            auto visible_indices = torch::nonzero(groups_per_gauss).to(torch::kInt32);
            cum_tiles_per_gauss = torch::cumsum(tiles_per_gauss.view({-1}), 0);
            cum_groups_per_gauss = torch::cumsum(groups_per_gauss.view({-1}), 0);
            cum_groups_per_gauss = torch::pad(cum_groups_per_gauss, {1, 0});

            n_isects = cum_tiles_per_gauss[-1].item<int64_t>();
            n_groups = cum_groups_per_gauss[-1].item<int64_t>();
        }
        else
        {
            n_isects = 0;
        }
        // printf("n_isects: %ld, n_groups: %ld\n", n_isects, n_groups);
        // cudaDeviceSynchronize();
        // clock_gettime(CLOCK_MONOTONIC, &first_pass_end);

        // cudaDeviceSynchronize();
        // clock_gettime(CLOCK_MONOTONIC, &second_pass_start);
        // second pass: compute isect_ids and flatten_ids as a packed tensor
        torch::Tensor isect_ids =
            torch::empty({n_isects}, depths.options().dtype(torch::kInt64));
        torch::Tensor group_gs_ids =
            torch::empty({n_groups}, depths.options().dtype(torch::kInt32));
        torch::Tensor group_starts =
            torch::empty({n_groups}, depths.options().dtype(torch::kInt32));
        torch::Tensor flatten_ids =
            torch::empty({n_isects}, depths.options().dtype(torch::kInt32));
        if (n_isects)
        {
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half,
                at::ScalarType::BFloat16,
                means2d.scalar_type(),
                "isect_tiles_n_isects",
                [&]()
                {
                    isect_tiles_no_depth<<<
                        (total_elems + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                        GSPLAT_N_THREADS,
                        0,
                        stream>>>(
                        packed,
                        C,
                        N,
                        nnz,
                        camera_ids_ptr,
                        gaussian_ids_ptr,
                        reinterpret_cast<scalar_t *>(means2d.data_ptr<scalar_t>()),
                        radii.data_ptr<int32_t>(),
                        depths.data_ptr<scalar_t>(),
                        cum_tiles_per_gauss.data_ptr<int64_t>(),
                        cum_groups_per_gauss.data_ptr<int64_t>(),
                        tile_size,
                        tile_width,
                        tile_height,
                        tile_n_bits,
                        nullptr,
                        groups_per_gauss.data_ptr<int32_t>(),
                        isect_ids.data_ptr<int64_t>(),
                        group_gs_ids.data_ptr<int32_t>(),
                        group_starts.data_ptr<int32_t>(),
                        flatten_ids.data_ptr<int32_t>());
                });
        }
        // cudaDeviceSynchronize();
        // clock_gettime(CLOCK_MONOTONIC, &second_pass_end);

        // cudaDeviceSynchronize();
        // clock_gettime(CLOCK_MONOTONIC, &sort_pass_start);
        // optionally sort the Gaussians by isect_ids
        if (n_isects && sort)
        {
            torch::Tensor isect_ids_sorted = torch::empty_like(isect_ids);
            torch::Tensor flatten_ids_sorted = torch::empty_like(flatten_ids);

            // https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html
            // DoubleBuffer reduce the auxiliary memory usage from O(N+P) to O(P)
            if (double_buffer)
            {
                // Create a set of DoubleBuffers to wrap pairs of device pointers
                cub::DoubleBuffer<int64_t> d_keys(
                    isect_ids.data_ptr<int64_t>(),
                    isect_ids_sorted.data_ptr<int64_t>());
                cub::DoubleBuffer<int32_t> d_values(
                    flatten_ids.data_ptr<int32_t>(),
                    flatten_ids_sorted.data_ptr<int32_t>());
                GSPLAT_CUB_WRAPPER(
                    cub::DeviceRadixSort::SortPairs,
                    d_keys,
                    d_values,
                    n_isects,
                    0,
                    // 32 + tile_n_bits + cam_n_bits,
                    tile_n_bits + cam_n_bits,
                    stream);
                switch (d_keys.selector)
                {
                case 0: // sorted items are stored in isect_ids
                    isect_ids_sorted = isect_ids;
                    break;
                case 1: // sorted items are stored in isect_ids_sorted
                    break;
                }
                switch (d_values.selector)
                {
                case 0: // sorted items are stored in flatten_ids
                    flatten_ids_sorted = flatten_ids;
                    break;
                case 1: // sorted items are stored in flatten_ids_sorted
                    break;
                }
            }
            else
            {
                GSPLAT_CUB_WRAPPER(
                    cub::DeviceRadixSort::SortPairs,
                    isect_ids.data_ptr<int64_t>(),
                    isect_ids_sorted.data_ptr<int64_t>(),
                    flatten_ids.data_ptr<int32_t>(),
                    flatten_ids_sorted.data_ptr<int32_t>(),
                    n_isects,
                    0,
                    tile_n_bits + cam_n_bits,
                    stream);
            }
            // cudaDeviceSynchronize();
            // clock_gettime(CLOCK_MONOTONIC, &sort_pass_end);
            // auto first_pass_time = double(first_pass_end.tv_nsec - first_pass_start.tv_nsec) / 1000000;
            // auto second_pass_time = double(second_pass_end.tv_nsec - second_pass_start.tv_nsec) / 1000000;
            // auto sort_time = double(sort_pass_end.tv_nsec - sort_pass_start.tv_nsec) / 1000000;
            // printf("first pass time: %f, second pass time: %f, sort time: %f, max group num: %d\n", first_pass_time, second_pass_time, sort_time, groups_per_gauss.max().item<int>());
            return std::make_tuple(
                tiles_per_gauss, isect_ids_sorted, flatten_ids_sorted, group_gs_ids, group_starts);
        }
        else
        {
            return std::make_tuple(tiles_per_gauss, isect_ids, flatten_ids, group_gs_ids, group_starts);
        }
    }

    __global__ void isect_offset_encode_no_depth(
        const uint32_t n_isects,
        const int64_t *__restrict__ isect_ids,
        const uint32_t C,
        const uint32_t n_tiles,
        const uint32_t tile_n_bits,
        int32_t *__restrict__ offsets // [C, n_tiles]
    )
    {
        // e.g., ids: [1, 1, 1, 3, 3], n_tiles = 6
        // counts: [0, 3, 0, 2, 0, 0]
        // cumsum: [0, 3, 3, 5, 5, 5]
        // offsets: [0, 0, 3, 3, 5, 5]
        uint32_t idx = cg::this_grid().thread_rank();
        if (idx >= n_isects)
            return;

        // int64_t isect_id_curr = isect_ids[idx] >> 32;
        int64_t isect_id_curr = isect_ids[idx];
        int64_t cid_curr = isect_id_curr >> tile_n_bits;
        int64_t tid_curr = isect_id_curr & ((1 << tile_n_bits) - 1);
        int64_t id_curr = cid_curr * n_tiles + tid_curr;

        if (idx == 0)
        {
            // write out the offsets until the first valid tile (inclusive)
            for (uint32_t i = 0; i < id_curr + 1; ++i)
                offsets[i] = static_cast<int32_t>(idx);
        }
        if (idx == n_isects - 1)
        {
            // write out the rest of the offsets
            for (uint32_t i = id_curr + 1; i < C * n_tiles; ++i)
                offsets[i] = static_cast<int32_t>(n_isects);
        }

        if (idx > 0)
        {
            // visit the current and previous isect_id and check if the (cid,
            // tile_id) pair changes.
            // int64_t isect_id_prev = isect_ids[idx - 1] >> 32; // shift out the depth
            int64_t isect_id_prev = isect_ids[idx - 1]; // shift out the depth
            if (isect_id_prev == isect_id_curr)
                return;

            // write out the offsets between the previous and current tiles
            int64_t cid_prev = isect_id_prev >> tile_n_bits;
            int64_t tid_prev = isect_id_prev & ((1 << tile_n_bits) - 1);
            int64_t id_prev = cid_prev * n_tiles + tid_prev;
            for (uint32_t i = id_prev + 1; i < id_curr + 1; ++i)
                offsets[i] = static_cast<int32_t>(idx);
        }
    }

    torch::Tensor isect_offset_encode_tensor_no_depth(
        const torch::Tensor &isect_ids, // [n_isects]
        const uint32_t C,
        const uint32_t tile_width,
        const uint32_t tile_height)
    {
        GSPLAT_DEVICE_GUARD(isect_ids);
        GSPLAT_CHECK_INPUT(isect_ids);

        uint32_t n_isects = isect_ids.size(0);
        torch::Tensor offsets = torch::empty(
            {C, tile_height, tile_width}, isect_ids.options().dtype(torch::kInt32));
        if (n_isects)
        {
            uint32_t n_tiles = tile_width * tile_height;
            uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
            at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
            isect_offset_encode_no_depth<<<
                (n_isects + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                GSPLAT_N_THREADS,
                0,
                stream>>>(
                n_isects,
                isect_ids.data_ptr<int64_t>(),
                C,
                n_tiles,
                tile_n_bits,
                offsets.data_ptr<int32_t>());
        }
        else
        {
            offsets.fill_(0);
        }
        return offsets;
    }

} // namespace gsplat