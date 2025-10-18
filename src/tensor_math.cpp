#include "tensor_math.h"

using namespace torch::indexing;

ORUtils::Matrix4<float> *tensorToInfiMatrix4(const torch::Tensor &tensor)
{
    // 确保 tensor 是 4x4 的
    assert(tensor.sizes() == c10::IntArrayRef({4, 4}));

    // 确保数据类型匹配（这里假设 T 是 float，如果是 double 需要相应调整）
    assert(tensor.dtype() == torch::kFloat32);

    // 创建 Matrix4 对象
    ORUtils::Matrix4<float> *matrix = new ORUtils::Matrix4<float>();

    // 填充 Matrix4
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            (*matrix)(j, i) = tensor[i][j].item<float>();
        }
    }
    return matrix;
}

torch::Tensor infiMatrix4ToTensor(const ORUtils::Matrix4<float> &matrix)
{
    torch::Tensor tensor = torch::zeros({4, 4});
    // 填充 Matrix4
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            tensor[i][j] = matrix(j, i);
        }
    }
    return tensor;
}

torch::Tensor l1(const torch::Tensor &rendered, const torch::Tensor &gt)
{
    return torch::abs(gt - rendered).mean();
}

torch::Tensor getR(const torch::Tensor &pose)
{
    return pose.index({Slice(None, 3), Slice(None, 3)});
}

torch::Tensor getT(const torch::Tensor &pose)
{
    return pose.index({Slice(None, 3), Slice(3, 4)});
}

torch::Tensor poseInv(const torch::Tensor &pose)
{
    // worldToCam
    torch::Tensor R = getR(pose);
    torch::Tensor T = getT(pose);
    torch::Tensor Rinv = R.transpose(0, 1);
    torch::Tensor Tinv = torch::matmul(-Rinv, T);
    torch::Tensor pose_inv = torch::eye(4, pose.options());
    pose_inv.index_put_({Slice(None, 3), Slice(None, 3)}, Rinv);
    pose_inv.index_put_({Slice(None, 3), Slice(3, 4)}, Tinv);
    return pose_inv;
}

torch::Tensor verticesTransform(const torch::Tensor &vertex_tensor, const torch::Tensor transform)
{
    int num_vertices = vertex_tensor.size(0) * vertex_tensor.size(1);
    torch::Tensor vertices = vertex_tensor.reshape({num_vertices, 3});
    torch::Tensor homogeneous_vertices = torch::ones({num_vertices, 4}, vertices.options());
    homogeneous_vertices.slice(1, 0, 3) = vertices;
    homogeneous_vertices = homogeneous_vertices.transpose(0, 1);
    torch::Tensor transformed_vertices = transform.matmul(homogeneous_vertices);
    transformed_vertices = transformed_vertices.transpose(0, 1);
    transformed_vertices = transformed_vertices.slice(1, 0, 3) / transformed_vertices.slice(1, 3, 4);
    transformed_vertices = transformed_vertices.reshape({vertex_tensor.size(0), vertex_tensor.size(1), 3});
    return transformed_vertices;
}

torch::Tensor quatToRotMat(const torch::Tensor &quat)
{
    auto u = torch::unbind(torch::nn::functional::normalize(quat, torch::nn::functional::NormalizeFuncOptions().dim(-1)), -1);
    torch::Tensor w = u[0];
    torch::Tensor x = u[1];
    torch::Tensor y = u[2];
    torch::Tensor z = u[3];
    return torch::stack({torch::stack({1.0 - 2.0 * (y.pow(2) + z.pow(2)),
                                       2.0 * (x * y - w * z),
                                       2.0 * (x * z + w * y)},
                                      -1),
                         torch::stack({2.0 * (x * y + w * z),
                                       1.0 - 2.0 * (x.pow(2) + z.pow(2)),
                                       2.0 * (y * z - w * x)},
                                      -1),
                         torch::stack({2.0 * (x * z - w * y),
                                       2.0 * (y * z + w * x),
                                       1.0 - 2.0 * (x.pow(2) + y.pow(2))},
                                      -1)},
                        -2);
}

std::tuple<torch::Tensor, torch::Tensor, float> autoScaleAndCenterPoses(const torch::Tensor &poses)
{
    // Center at mean
    torch::Tensor origins = poses.index({"...", Slice(None, 3), 3});
    torch::Tensor center = torch::mean(origins, 0);
    origins -= center;

    // Scale
    float f = 1.0f / torch::max(torch::abs(origins)).item<float>();
    origins *= f;

    torch::Tensor transformedPoses = poses.clone();
    transformedPoses.index_put_({"...", Slice(None, 3), 3}, origins);

    return std::make_tuple(transformedPoses, center, f);
}

torch::Tensor rotationMatrix(const torch::Tensor &a, const torch::Tensor &b)
{
    // Rotation matrix that rotates vector a to vector b
    torch::Tensor a1 = a / a.norm();
    torch::Tensor b1 = b / b.norm();
    torch::Tensor v = torch::linalg_cross(a1, b1);
    torch::Tensor c = torch::dot(a1, b1);
    const float EPS = 1e-8;
    if (c.item<float>() < -1 + EPS)
    {
        torch::Tensor eps = (torch::rand(3) - 0.5f) * 0.01f;
        return rotationMatrix(a1 + eps, b1);
    }
    torch::Tensor s = v.norm();
    torch::Tensor skew = torch::zeros({3, 3}, torch::kFloat32);
    skew[0][1] = -v[2];
    skew[0][2] = v[1];
    skew[1][0] = v[2];
    skew[1][2] = -v[0];
    skew[2][0] = -v[1];
    skew[2][1] = v[0];

    return torch::eye(3) + skew + torch::matmul(skew, skew * ((1 - c) / (s.pow(2) + EPS)));
}

torch::Tensor rodriguesToRotation(const torch::Tensor &rodrigues)
{
    float theta = torch::linalg_vector_norm(rodrigues, 2, {-1}, true, torch::kFloat32).item<float>();
    if (theta < 1e-9f)
    {
        return torch::eye(3, torch::kFloat32);
    }
    torch::Tensor r = rodrigues / theta;
    torch::Tensor ident = torch::eye(3, torch::kFloat32);
    float a = r[0].item<float>();
    float b = r[1].item<float>();
    float c = r[2].item<float>();
    torch::Tensor rrT = torch::tensor({{a * a, a * b, a * c},
                                       {b * a, b * b, b * c},
                                       {c * a, c * b, c * c}},
                                      torch::kFloat32);
    torch::Tensor rCross = torch::tensor({{0.0f, -c, b},
                                          {c, 0.0f, -a},
                                          {-b, a, 0.0f}},
                                         torch::kFloat32);
    float cosTheta = std::cos(theta);

    return cosTheta * ident + (1 - cosTheta) * rrT + std::sin(theta) * rCross;
}

torch::Tensor randomQuatTensor(long long n)
{
    torch::Tensor u = torch::rand(n);
    torch::Tensor v = torch::rand(n);
    torch::Tensor w = torch::rand(n);
    return torch::stack({torch::sqrt(1 - u) * torch::sin(2 * PI * v),
                         torch::sqrt(1 - u) * torch::cos(2 * PI * v),
                         torch::sqrt(u) * torch::sin(2 * PI * w),
                         torch::sqrt(u) * torch::cos(2 * PI * w)},
                        -1);
}

torch::Tensor computeQuat(const torch::Tensor &init_vec, const torch::Tensor &target_vec)
{
    torch::Tensor axis = torch::cross(init_vec, target_vec, 1);
    axis = axis / (torch::norm(axis, 2, -1, true) + 1e-8);
    torch::Tensor angle = torch::acos(torch::sum(init_vec * target_vec, 1)).unsqueeze(-1);
    torch::Tensor quat = quaternionFromAxisAngle(axis, angle);
    return quat;
}

torch::Tensor quaternionFromAxisAngle(const torch::Tensor &axis, const torch::Tensor &angle)
{
    torch::Tensor normalized_axis = axis / (torch::norm(axis, 2, -1, true) + 1e-8);
    torch::Tensor half_angle = angle / 2;
    torch::Tensor real_part = torch::cos(half_angle).toType(axis.scalar_type());
    torch::Tensor complex_part = normalized_axis * torch::sin(half_angle).toType(axis.scalar_type());
    torch::Tensor quaternion = torch::cat({real_part, complex_part}, 1);
    return quaternion;
}

torch::Tensor projectionMatrix(float zNear, float zFar, float fovX, float fovY, const torch::Device &device)
{
    // OpenGL perspective projection matrix
    float t = zNear * std::tan(0.5f * fovY);
    float b = -t;
    float r = zNear * std::tan(0.5f * fovX);
    float l = -r;
    return torch::tensor({{2.0f * zNear / (r - l), 0.0f, (r + l) / (r - l), 0.0f},
                          {0.0f, 2 * zNear / (t - b), (t + b) / (t - b), 0.0f},
                          {0.0f, 0.0f, (zFar + zNear) / (zFar - zNear), -1.0f * zFar * zNear / (zFar - zNear)},
                          {0.0f, 0.0f, 1.0f, 0.0f}},
                         device);
}

std::tuple<torch::Tensor, torch::Tensor> featureGradient(const torch::Tensor &img, bool normalize_gradient)
{
    auto H = img.size(0);
    auto W = img.size(1);
    auto C = img.size(2);

    auto options = torch::TensorOptions().device(img.device()).dtype(img.dtype());

    // Sobel filters
    auto wx = torch::tensor({{{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}}}, options).view({1, 1, 3, 3});
    auto wy = torch::tensor({{{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}}}, options).view({1, 1, 3, 3});

    // Permute and reshape image
    auto img_permuted = img.permute({2, 0, 1}).view({-1, 1, H, W}); // [c, 1, h, w]
    // std::cout << img.permute({2, 0, 1}).sizes() << std::endl;
    // std::cout << img_permuted.sizes() << std::endl;

    // Pad the image
    auto img_pad = torch::nn::functional::pad(img_permuted, torch::nn::functional::PadFuncOptions({1, 1, 1, 1}).mode(torch::kReplicate));
    // Convolve with Sobel filters
    auto img_dx = torch::nn::functional::conv2d(img_pad, wx).squeeze().permute({1, 2, 0}); // [h, w, c]
    auto img_dy = torch::nn::functional::conv2d(img_pad, wy).squeeze().permute({1, 2, 0}); // [h, w, c]

    if (normalize_gradient)
    {
        auto mag = torch::sqrt(img_dx.pow(2) + img_dy.pow(2) + 1e-8);
        img_dx = img_dx / mag;
        img_dy = img_dy / mag;
    }

    return std::make_tuple(img_dx, img_dy);
}

torch::Tensor computeVertexMap(const torch::Tensor &depth_map, const torch::Tensor intrinsics)
{
    int H = depth_map.size(0);
    int W = depth_map.size(1);

    auto fx = intrinsics[0][0].item<float>();
    auto fy = intrinsics[1][1].item<float>();
    auto cx = intrinsics[0][2].item<float>();
    auto cy = intrinsics[1][2].item<float>();

    auto options = torch::TensorOptions().device(depth_map.device()).dtype(depth_map.dtype());

    // 生成网格
    auto i = torch::linspace(0, W - 1, W, options);
    auto j = torch::linspace(0, H - 1, H, options);
    auto meshgrid = torch::meshgrid({j, i}, "ij");
    auto i_grid = meshgrid[1]; // [H, W]
    auto j_grid = meshgrid[0]; // [H, W]

    // 计算顶点图
    auto x = (i_grid - cx) / fx;       // [H, W, 1]
    auto y = (j_grid - cy) / fy;       // [H, W, 1]
    auto z = torch::ones_like(i_grid); // [H, W, 1]

    auto vertex_map_c = torch::stack({x, y, z}, -1) * depth_map; // [H, W, 3]
    return vertex_map_c;
}

torch::Tensor computeNormalMap(const torch::Tensor &vertex_map)
{
    auto H = vertex_map.size(0);
    auto W = vertex_map.size(1);

    auto [img_dx, img_dy] = featureGradient(vertex_map, false);
    // std::cout << img_dx.sizes() << std::endl;
    // std::cout << img_dy.sizes() << std::endl;

    auto normal = torch::cross(img_dy.view({-1, 3}), img_dx.view({-1, 3}), -1);
    normal = normal.view({H, W, 3});

    auto mag = torch::norm(normal, 2, -1, true);
    normal = normal / (mag + 1e-8);

    // Filter out invalid pixels
    auto depth = vertex_map.index({torch::indexing::Slice(), torch::indexing::Slice(), 2});
    auto invalid_mask = depth <= 0;
    auto zero_normal = torch::zeros_like(normal);
    normal = torch::where(invalid_mask.unsqueeze(-1), zero_normal, normal);

    return normal;
}

float rotCompare(const torch::Tensor &prev_rot, const torch::Tensor &curr_rot)
{
    // 计算旋转差异
    auto rot_diff = torch::matmul(prev_rot.transpose(0, 1), curr_rot);

    // 计算 cos_theta
    auto cos_theta = (rot_diff.trace() - 1) / 2;

    // 计算弧度差
    auto rad_diff = torch::acos(cos_theta);

    // 计算角度差（将弧度转换为角度）
    auto theta_diff = rad_diff * 180 / PI;

    return theta_diff.item<float>();
}

float transCompare(const torch::Tensor &prev_trans, const torch::Tensor &curr_trans)
{
    // 计算平移差异
    auto trans_diff = prev_trans - curr_trans;

    // 计算L2范数差异
    auto l2_diff = torch::norm(trans_diff, 2);

    return l2_diff.item<float>();
}