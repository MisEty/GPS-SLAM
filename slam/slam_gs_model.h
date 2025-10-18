#include "raw_gs_model.h"

class SLAMGaussianModel : public RawGaussianModel
{
public:
    void addGaussians(const Camera &cam,
                      TensorDict &frame_maps,
                      const torch::Tensor &sample_mask,
                      float new_gs_sample_ratio,
                      int frame_num);
};