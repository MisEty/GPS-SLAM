#include "kdtree_tensor.h"

PointsTensor::~PointsTensor()
{
    freeIndex<KdTreeTensor>();
}

torch::Tensor PointsTensor::scales(float max_scale, float min_scale, int nearest_count)
{
    // Compute scales by finding the average of the three nearest neighbors for each point
    const auto index = getIndex<KdTreeTensor>();
    torch::Tensor scales = torch::zeros({static_cast<long int>(tensor.size(0)), 1}, torch::kFloat32);
    int count = nearest_count + 1;

    std::vector<size_t> indices(count);
    std::vector<float> sqr_dists(count);
    //     omp_set_num_threads(32);
    // #pragma omp parallel for
    for (size_t i = 0; i < tensor.size(0); i++)
    {
        // if (i % 1000 == 0)
        //     std::cout << i << std::endl;
        index->knnSearch(reinterpret_cast<float *>(tensor[i].data_ptr()), count, indices.data(), sqr_dists.data());
        float sum = 0.0;
        for (size_t j = 1; j < count; j++)
        {
            if (i == 0)
                std::cout << sqr_dists[j] << std::endl;;
            sum += sqr_dists[j];
        }
        float mean_dist = sum / (count - 1);
        if (max_scale > 0)
            mean_dist = std::min(mean_dist, max_scale * max_scale);
        if (min_scale > 0)
            mean_dist = std::max(mean_dist, min_scale * min_scale);
        scales[i] = std::sqrt(mean_dist);
    }
    return scales;
}