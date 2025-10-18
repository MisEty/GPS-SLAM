#include "cv_utils.h"

// 把cv mat转换为torch tensor
torch::Tensor matToTensor(const cv::Mat &mat)
{
    torch::Tensor tensor;

    // 检查 Mat 的类型并相应地处理
    switch (mat.type())
    {
    case CV_8UC1:
    case CV_8UC3:
    case CV_8UC4:
    {
        // 对于 uchar 类型，我们需要先转换为 float 并归一化
        cv::Mat matFloat;
        mat.convertTo(matFloat, CV_32F, 1.0 / 255.0);
        tensor = torch::from_blob(matFloat.data,
                                  {matFloat.rows, matFloat.cols, matFloat.channels()},
                                  torch::kFloat32);
        break;
    }
    case CV_32FC1:
    case CV_32FC3:
    case CV_32FC4:
    {
        // 对于 float 类型，我们可以直接转换
        tensor = torch::from_blob(mat.data,
                                  {mat.rows, mat.cols, mat.channels()},
                                  torch::kFloat32);
        break;
    }
    default:
        throw std::runtime_error("Unsupported Mat type");
    }

    // 返回一个新的 tensor，而不是对原始数据的视图
    return tensor.clone();
}

cv::Mat imreadRGB(const std::string &filename)
{
    cv::Mat cImg = cv::imread(filename);

    if (cImg.empty())
    {
        std::cerr << "Cannot read " << filename << std::endl
                  << "Make sure the path to your images is correct" << std::endl;
        exit(1);
    }

    cv::cvtColor(cImg, cImg, cv::COLOR_BGR2RGB);
    return cImg;
}

// 把float类型的颜色tensor转换成0-255的cv mat
cv::Mat tensorToImage(const torch::Tensor &t)
{
    torch::Tensor tensor = t.detach().cpu();
    int h = tensor.sizes()[0];
    int w = tensor.sizes()[1];
    int c = tensor.sizes()[2];

    int type = CV_8UC3;
    if (c != 3)
        throw std::runtime_error("Only images with 3 channels are supported");

    cv::Mat image(h, w, type);
    torch::Tensor scaled_tensor = (tensor * 255.0).toType(torch::kU8);

    torch::Tensor cont_tensor = scaled_tensor.contiguous();
    std::memcpy(image.data, cont_tensor.data_ptr<uint8_t>(), h * w * c * sizeof(uint8_t));

    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    return image;
}

// 把float类型的深度tensor转换成short类型的的cv mat
cv::Mat tensorToDepth(const torch::Tensor &t)
{
    torch::Tensor tensor = t.detach().cpu();
    int h = tensor.sizes()[0];
    int w = tensor.sizes()[1];
    int c = tensor.sizes()[2];

    int type = CV_32FC1;
    if (c != 1)
        throw std::runtime_error("Only images with 1 channels are supported");

    cv::Mat image(h, w, type);

    // 创建一个新的连续内存的 tensor
    torch::Tensor cont_tensor = tensor.contiguous();

    // 复制数据到 OpenCV Mat
    std::memcpy(image.data, cont_tensor.data_ptr<float>(), h * w * c * sizeof(float));

    // 将 float Mat 转换为 short Mat
    cv::Mat depth_image;
    image.convertTo(depth_image, CV_16UC1, 1000); // 转换成以mm为单位
    return depth_image;
}

cv::Mat imageToJetMat(const cv::Mat &depth_image, float min_depth, float max_depth)
{
    cv::Mat normalized;

    // 步骤1：归一化到0-255范围
    depth_image.convertTo(normalized, CV_32F, 255.0 / (max_depth - min_depth), -min_depth * 255.0 / (max_depth - min_depth));
    // 步骤2：转换为8位无符号整型
    cv::Mat depth_8u;
    normalized.convertTo(depth_8u, CV_8U);
    // 步骤3：应用jet颜色映射
    cv::Mat output;
    cv::applyColorMap(depth_8u, output, cv::COLORMAP_JET);

    // 可选：处理超出范围的值
    cv::Mat mask = depth_image < min_depth;
    output.setTo(cv::Scalar(0, 0, 0), mask); // 将小于min_depth的区域设为黑色

    mask = depth_image > max_depth;
    output.setTo(cv::Scalar(255, 255, 255), mask); // 将大于max_depth的区域设为白色
    return output;
}

// 把一个tensor使用Jet colormap可视化为cv Mat
cv::Mat tensorToJetMat(const torch::Tensor &tensor_input, float min, float max, bool set_zero)
{
    torch::Tensor tensor = tensor_input.detach().cpu();
    cv::Mat tensor_mat_raw(tensor.size(0), tensor.size(1), CV_32F, tensor.data_ptr());

    // 归一化tensor
    torch::Tensor normalized = (tensor - min) / (max - min);
    normalized = normalized.clamp(0, 1); // 确保值在[0,1]范围内

    // 将归一化的tensor转换为OpenCV的Mat对象
    cv::Mat tensor_mat(normalized.size(0), normalized.size(1), CV_32F, normalized.data_ptr());

    // 将mat转换为彩色图像 - 使用新的矩阵作为目标
    cv::Mat tensor_mat_8bit;                               // 新增一个矩阵作为转换目标
    tensor_mat.convertTo(tensor_mat_8bit, CV_8UC1, 255.0); // 转换为8位无符号整数

    cv::Mat color_map;
    cv::applyColorMap(tensor_mat_8bit, color_map, cv::COLORMAP_JET);

    // 将小于0的像素设置为黑色
    cv::Mat neg_mask;
    cv::compare(tensor_mat_raw, 0, neg_mask, cv::CMP_LT); // 正确创建二值掩码
    color_map.setTo(cv::Scalar(0, 0, 0), neg_mask);

    if (set_zero)
    {
        // 将值为0的像素设置为黑色
        cv::Mat zero_mask;
        cv::compare(tensor_mat_8bit, 0, zero_mask, cv::CMP_EQ);
        color_map.setTo(cv::Scalar(0, 0, 0), zero_mask);
    }
    cv::cvtColor(color_map, color_map, cv::COLOR_RGB2BGR);

    return color_map;
}

// 比较两个颜色图像并拼接误差图
cv::Mat colorCompImg(const torch::Tensor &render_rgb,
                     const torch::Tensor &gt_rgb,
                     float min_error_thres,
                     float max_error_thres)
{
    auto rgb = render_rgb.detach().cpu();
    auto gt = gt_rgb.detach().cpu();
    torch::Tensor color_compare = torch::cat({gt, rgb}, 0);
    cv::Mat rgb_image = tensorToImage(color_compare);

    // 计算颜色误差
    torch::Tensor color_error = torch::abs(rgb - gt).mean(2);
    // 在颜色可视化的时候，0不代表无效值，所以不需要设为黑色
    cv::Mat color_error_img = tensorToJetMat(color_error, min_error_thres, max_error_thres, false);
    cv::cvtColor(rgb_image, rgb_image, cv::COLOR_RGB2BGR);
    cv::vconcat(rgb_image, color_error_img, rgb_image);
    cv::cvtColor(rgb_image, rgb_image, cv::COLOR_RGB2BGR);
    return rgb_image;
}

// 比较两个深度图像并拼接误差图
cv::Mat depthCompImg(const torch::Tensor &render_depth,
                     const torch::Tensor &gt_depth,
                     float min_vis_depth, float max_vis_depth,
                     float min_error_thres, float max_error_thres)
{
    auto depth = render_depth.detach().cpu();
    auto gt = gt_depth.detach().cpu();
    torch::Tensor invalid_depth = depth.eq(0) | gt.eq(0);

    // 在深度可视化的时候，0代表无效值，需要设为黑色
    cv::Mat depth_image = tensorToJetMat(depth, min_vis_depth, max_vis_depth, true);
    cv::Mat gt_depth_image = tensorToJetMat(gt, min_vis_depth, max_vis_depth, true);

    torch::Tensor depth_error = torch::abs(depth - gt) + 1e-3;
    depth_error.masked_fill_(invalid_depth, 0); // 把render和gt的无效像素都设0
    // std::cout << min_error_thres << "\t" << max_error_thres << std::endl;
    cv::Mat depth_error_image = tensorToJetMat(depth_error, min_error_thres, max_error_thres, true);
    cv::vconcat(gt_depth_image, depth_image, depth_image);
    cv::vconcat(depth_image, depth_error_image, depth_image);
    cv::cvtColor(depth_image, depth_image, cv::COLOR_RGB2BGR);
    return depth_image;
}

torch::Tensor imageToTensor(const cv::Mat &image)
{
    torch::Tensor img = torch::from_blob(image.data, {image.rows, image.cols, image.channels()}, torch::kU8);
    torch::Tensor res = img.toType(torch::kFloat32) / 255.0f;
    return res.clone();
}

torch::Tensor depthToTensor(const cv::Mat &depth)
{
    torch::Tensor dep = torch::from_blob(depth.data, {depth.rows, depth.cols, depth.channels()}, torch::kFloat32);
    return dep.clone();
}

ITMUChar4Image *imageToITMUChar4Image(const cv::Mat &image, MemoryDeviceType type)
{
    int width = image.cols;
    int height = image.rows;
    int channels = image.channels();

    // 创建 Image<Vector4u>
    ITMUChar4Image *result = new ITMUChar4Image(ORUtils::Vector2<int>(width, height), true, false); // 只在 CPU 上分配内存
    ORUtils::Vector4<unsigned char> *dataPtr = result->GetData(type);

    // 遍历每个像素
    int count = 0;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            cv::Vec3b bgr = image.at<cv::Vec3b>(y, x);
            dataPtr[count].x = bgr[2];
            dataPtr[count].y = bgr[1];
            dataPtr[count].z = bgr[0];
            dataPtr[count].w = 255; // 设置 alpha 为不透明
            count++;
        }
    }
    return result;
}

ITMShortImage *depthToITMUShortImage(const cv::Mat &depth, MemoryDeviceType type)
{
    int width = depth.cols;
    int height = depth.rows;

    // 创建 Image<short>
    ITMShortImage *result = new ITMShortImage(ORUtils::Vector2<int>(width, height), true, false); // 只在 CPU 上分配内存
    short *dataPtr = result->GetData(type);

    // 遍历每个像素
    int count = 0;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            dataPtr[count] = depth.at<short>(y, x);
            count++;
        }
    }
    return result;
}

cv::Mat ITMUChar4ImageToImage(const ITMUChar4Image *itmu_image, MemoryDeviceType type)
{
    if (itmu_image == nullptr)
    {
        return cv::Mat();
    }

    // 获取图像尺寸
    int width = itmu_image->noDims.x;
    int height = itmu_image->noDims.y;

    // 创建 cv::Mat
    cv::Mat result(height, width, CV_8UC4);

    // 获取 ITMUChar4Image 的数据指针
    const ORUtils::Vector4<unsigned char> *dataPtr = itmu_image->GetData(type);

    // 遍历每个像素
    memcpy(result.data, dataPtr, width * height * sizeof(ORUtils::Vector4<unsigned char>));

    // 取三通道
    cv::Mat dst(result.rows, result.cols, CV_8UC3);
    int from_to[] = {0, 0, 1, 1, 2, 2};
    cv::mixChannels(&result, 1, &dst, 1, from_to, 3);
    return dst;
}

cv::Mat ITMUFloat4ImageToImage(const ITMFloat4Image *itmu_image, MemoryDeviceType type)
{
    if (itmu_image == nullptr)
    {
        return cv::Mat();
    }

    // 获取图像尺寸
    int width = itmu_image->noDims.x;
    int height = itmu_image->noDims.y;

    // 创建 cv::Mat
    cv::Mat result(height, width, CV_32FC4);

    // 获取 ITMUChar4Image 的数据指针
    const ORUtils::Vector4<float> *dataPtr = itmu_image->GetData(type);
    memcpy(result.data, dataPtr, width * height * sizeof(ORUtils::Vector4<float>));

    // 取三通道
    cv::Mat dst(result.rows, result.cols, CV_32FC3);
    int from_to[] = {0, 0, 1, 1, 2, 2};
    cv::mixChannels(&result, 1, &dst, 1, from_to, 3);
    return dst;
}

torch::Tensor ITMUChar4ImageToTensor(ITMUChar4Image *itmu_image, MemoryDeviceType type)
{
    ORUtils::Vector4<unsigned char> *dataPtr = itmu_image->GetData(type);
    int width = itmu_image->noDims.x;
    int height = itmu_image->noDims.y;
    torch::Tensor tensor = torch::from_blob(dataPtr, {height, width, 4}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA)).clone().to(torch::kFloat).div(255.0).slice(2, 0, 3);
    return tensor.contiguous();
}

torch::Tensor ITMUFloat4ImageToTensor(ITMFloat4Image *itmu_image, MemoryDeviceType type)
{
    ORUtils::Vector4<float> *dataPtr = itmu_image->GetData(type);
    int width = itmu_image->noDims.x;
    int height = itmu_image->noDims.y;
    torch::Tensor tensor = torch::from_blob(dataPtr, {height, width, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).clone();
    torch::Tensor value = tensor.slice(2, 0, 3);
    torch::Tensor confidence = tensor.slice(2, 3, 4);
    torch::Tensor valid_value = value * confidence.gt(0);
    torch::Tensor output = torch::cat({valid_value, confidence}, 2);
    return output.contiguous();
}