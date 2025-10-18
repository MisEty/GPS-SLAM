#include "gsplat_wapper.hpp"

double getDuration(struct timespec start, struct timespec end)
{
    // 计算秒和纳秒的时间差
    time_t seconds_diff = end.tv_sec - start.tv_sec;
    long nanoseconds_diff = end.tv_nsec - start.tv_nsec;

    // 将时间差转换为毫秒
    double interval_ms = (seconds_diff * 1000.0) + (nanoseconds_diff / 1000000.0);

    return interval_ms;
}

variable_list isectTiles(torch::Tensor means2d,
                         torch::Tensor radii,
                         torch::Tensor depths,
                         int tile_size,
                         int tile_width,
                         int tile_height,
                         bool sort)
{
    int C = means2d.size(0);
    int N = means2d.size(1);
    at::optional<torch::Tensor> camera_ids;
    at::optional<torch::Tensor> gaussian_ids;
    auto t = gsplat::isect_tiles_tensor(
        means2d.contiguous(),
        radii.contiguous(),
        depths.contiguous(),
        camera_ids,
        gaussian_ids,
        C,
        tile_size,
        tile_width,
        tile_height,
        sort,
        true);
    torch::Tensor tiles_per_gauss = std::get<0>(t);
    torch::Tensor isect_ids = std::get<1>(t);
    torch::Tensor flatten_ids = std::get<2>(t);
    return {tiles_per_gauss, isect_ids, flatten_ids};
}

torch::Tensor isectOffsetEncode(torch::Tensor isect_ids, int n_cameras, int tile_width, int tile_height)
{
    return gsplat::isect_offset_encode_tensor(isect_ids.contiguous(), n_cameras, tile_width, tile_height);
}

torch::Tensor simpleKNN(torch::Tensor points)
{
    return distCUDA2(points);
}

variable_list isectTilesNoDepth(torch::Tensor means2d,
                                torch::Tensor radii,
                                torch::Tensor depths,
                                int tile_size,
                                int tile_width,
                                int tile_height,
                                bool sort)
{
    int C = means2d.size(0);
    int N = means2d.size(1);
    at::optional<torch::Tensor> camera_ids;
    at::optional<torch::Tensor> gaussian_ids;
    auto t = gsplat::isect_tiles_tensor_no_depth(
        means2d.contiguous(),
        radii.contiguous(),
        depths.contiguous(),
        camera_ids,
        gaussian_ids,
        C,
        tile_size,
        tile_width,
        tile_height,
        sort,
        true);
    torch::Tensor tiles_per_gauss = std::get<0>(t);
    torch::Tensor isect_ids = std::get<1>(t);
    torch::Tensor flatten_ids = std::get<2>(t);
    torch::Tensor group_gs_ids = std::get<3>(t);
    torch::Tensor group_starts = std::get<4>(t);

    return {tiles_per_gauss, isect_ids, flatten_ids, group_gs_ids, group_starts};
}

torch::Tensor isectOffsetEncodeNoDepth(torch::Tensor isect_ids, int n_cameras, int tile_width, int tile_height)
{
    return gsplat::isect_offset_encode_tensor_no_depth(isect_ids.contiguous(), n_cameras, tile_width, tile_height);
}

int degFromSh(int numBases)
{
    switch (numBases)
    {
    case 1:
        return 0;
    case 4:
        return 1;
    case 9:
        return 2;
    case 16:
        return 3;
    default:
        return 4;
    }
}

int numShBases(int degree)
{
    switch (degree)
    {
    case 0:
        return 1;
    case 1:
        return 4;
    case 2:
        return 9;
    case 3:
        return 16;
    default:
        return 25;
    }
}

const double C0 = 0.28209479177387814;

torch::Tensor rgb2sh(const torch::Tensor &rgb)
{
    // Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    return (rgb - 0.5) / C0;
}

torch::Tensor sh2rgb(const torch::Tensor &sh)
{
    // Converts from 0th spherical harmonic coefficients to RGB values [0,1]
    return torch::clamp((sh * C0) + 0.5, 0.0f, 1.0f);
}