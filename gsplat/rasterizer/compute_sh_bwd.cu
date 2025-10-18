#include "bindings.h"
#include "spherical_harmonics.cuh"
#include "types.cuh"

#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace gsplat
{

    namespace cg = cooperative_groups;

    template <typename T>
    __global__ void compute_sh_bwd_kernel(
        const uint32_t N,
        const uint32_t K,
        const uint32_t degrees_to_use,
        const vec3<T> *__restrict__ dirs, // [N, 3]
        const T *__restrict__ coeffs,     // [N, K, 3]
        const bool *__restrict__ masks,   // [N]
        const T *__restrict__ v_colors,   // [N, 3
        T *__restrict__ v_coeffs,         // [N, K, 3]
        T *__restrict__ v_dirs            // [N, 3] optional
    )
    {
        // parallelize over N * 3
        uint32_t idx = cg::this_grid().thread_rank();
        if (idx >= N * 3)
        {
            return;
        }
        uint32_t elem_id = idx / 3;
        uint32_t c = idx % 3; // color channel
        if (masks != nullptr && !masks[elem_id])
        {
            return;
        }

        vec3<T> v_dir = {0.f, 0.f, 0.f};
        sh_coeffs_to_color_fast_vjp(
            degrees_to_use,
            c,
            dirs[elem_id],
            coeffs + elem_id * K * 3,
            v_colors + elem_id * 3,
            v_coeffs + elem_id * K * 3,
            v_dirs == nullptr ? nullptr : &v_dir);
        if (v_dirs != nullptr)
        {
            gpuAtomicAdd(v_dirs + elem_id * 3, v_dir.x);
            gpuAtomicAdd(v_dirs + elem_id * 3 + 1, v_dir.y);
            gpuAtomicAdd(v_dirs + elem_id * 3 + 2, v_dir.z);
        }
    }

    std::tuple<torch::Tensor, torch::Tensor> compute_sh_bwd_tensor(
        const uint32_t K,
        const uint32_t degrees_to_use,
        const torch::Tensor &dirs,               // [..., 3]
        const torch::Tensor &coeffs,             // [..., K, 3]
        const at::optional<torch::Tensor> masks, // [...]
        const torch::Tensor &v_colors,           // [..., 3]
        bool compute_v_dirs)
    {
        // struct timespec debug0, debug1, debug2, debug3, debug4;
        // cudaDeviceSynchronize();
        // clock_gettime(CLOCK_MONOTONIC, &debug0);
        GSPLAT_DEVICE_GUARD(dirs);
        GSPLAT_CHECK_INPUT(dirs);
        GSPLAT_CHECK_INPUT(coeffs);
        GSPLAT_CHECK_INPUT(v_colors);
        if (masks.has_value())
        {
            GSPLAT_CHECK_INPUT(masks.value());
        }
        TORCH_CHECK(v_colors.size(-1) == 3, "v_colors must have last dimension 3");
        TORCH_CHECK(coeffs.size(-1) == 3, "coeffs must have last dimension 3");
        TORCH_CHECK(dirs.size(-1) == 3, "dirs must have last dimension 3");
        // torch::Tensor v_coeffs = torch::zeros_like(coeffs);

        // cudaDeviceSynchronize();
        // clock_gettime(CLOCK_MONOTONIC, &debug1);

        const uint32_t N = dirs.numel() / 3;
        // std::cout << coeffs.sizes() << std::endl;
        torch::Tensor v_coeffs = torch::empty_like(coeffs).zero_();
        // cudaDeviceSynchronize();
        // clock_gettime(CLOCK_MONOTONIC, &debug2);
        torch::Tensor v_dirs;
        if (compute_v_dirs)
        {
            v_dirs = torch::zeros_like(dirs);
        }
        if (N)
        {
            // cudaDeviceSynchronize();
            // clock_gettime(CLOCK_MONOTONIC, &debug3);
            compute_sh_bwd_kernel<float>
                <<<(N * 3 + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                   GSPLAT_N_THREADS>>>(
                    N,
                    K,
                    degrees_to_use,
                    reinterpret_cast<vec3<float> *>(dirs.data_ptr<float>()),
                    coeffs.data_ptr<float>(),
                    masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
                    v_colors.data_ptr<float>(),
                    v_coeffs.data_ptr<float>(),
                    compute_v_dirs ? v_dirs.data_ptr<float>() : nullptr);
            // cudaDeviceSynchronize();
            // clock_gettime(CLOCK_MONOTONIC, &debug4);
            // auto debug1_time = double(debug1.tv_nsec - debug0.tv_nsec) / 1000000 + double(debug1.tv_sec - debug0.tv_sec) * 1000.0;

            // auto debug2_time = double(debug2.tv_nsec - debug1.tv_nsec) / 1000000 + double(debug2.tv_sec - debug1.tv_sec) * 1000.0;

            // auto debug3_time = double(debug3.tv_nsec - debug2.tv_nsec) / 1000000 + double(debug3.tv_sec - debug2.tv_sec) * 1000.0;

            // auto debug4_time = double(debug4.tv_nsec - debug3.tv_nsec) / 1000000 + double(debug4.tv_sec - debug3.tv_sec) * 1000.0;

            // printf("debug1 time: %f, debug2 time: %f, debug3 time: %f, debug4 time: %f\n", debug1_time, debug2_time, debug3_time, debug4_time);
        }
        return std::make_tuple(v_coeffs, v_dirs); // [..., K, 3], [..., 3]
    }
} // namespace gsplat
