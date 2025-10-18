#include <tinyply.h>

#include "dataset_reader.h"
#include "cv_utils.h"
#include "file_utils.h"
#include "tensor_math.h"

namespace fs = std::filesystem;
using namespace torch::indexing;
using json = nlohmann::json;

void Points::readPly(std::string filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    tinyply::PlyFile ply_file;
    ply_file.parse_header(file);

    std::shared_ptr<tinyply::PlyData> raw_xyzs, raw_colors, raw_normals;

    try
    {
        raw_xyzs = ply_file.request_properties_from_element("vertex", {"x", "y", "z"});
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error("Failed to read vertex positions: " + std::string(e.what()));
    }

    try
    {
        raw_colors = ply_file.request_properties_from_element("vertex", {"red", "green", "blue"});
    }
    catch (const std::exception &)
    {
        // Color data not available, ignore
    }

    try
    {
        raw_normals = ply_file.request_properties_from_element("vertex", {"nx", "ny", "nz"});
    }
    catch (const std::exception &)
    {
        // Normal data not available, ignore
    }

    ply_file.read(file);
    std::cout << "load points num: " << raw_xyzs->count << std::endl;
    // Convert raw_xyzs to torch tensor
    xyz = torch::from_blob(raw_xyzs->buffer.get(),
                           {static_cast<long>(raw_xyzs->count), 3}, torch::kFloat)
              .clone();

    // Convert raw_colors to torch tensor if available
    if (raw_colors)
    {
        rgb = torch::from_blob(raw_colors->buffer.get(),
                               {static_cast<long>(raw_colors->count), 3}, torch::kUInt8)
                  .to(torch::kFloat32)
                  .div(255.0)
                  .clone();
    }
    else
    {
        rgb = torch::Tensor();
    }

    // Convert raw_normals to torch tensor if available
    if (raw_normals)
    {
        normal = torch::from_blob(raw_normals->buffer.get(),
                                  {static_cast<long>(raw_normals->count), 3}, torch::kFloat)
                     .clone();
    }
    else
    {
        normal = torch::Tensor();
    }
}

torch::Tensor Camera::loadImage(float downscale_factor)
{
    if (image.numel())
        std::runtime_error("loadImage already called");

    cv::Mat cImg = imreadRGB(imgFilePath);
    if (downscale_factor > 1.0f)
    {
        float scaleFactor = 1.0f / downscale_factor;
        cv::resize(cImg, cImg, cv::Size(), scaleFactor, scaleFactor, cv::INTER_AREA);
        fx *= scaleFactor;
        fy *= scaleFactor;
        cx *= scaleFactor;
        cy *= scaleFactor;
    }
    torch::Tensor res = imageToTensor(cImg);

    K = torch::tensor({{fx, 0.0f, cx},
                       {0.0f, fy, cy},
                       {0.0f, 0.0f, 1.0f}},
                      torch::kFloat32);
    // Update parameters
    height = res.size(0);
    width = res.size(1);
    fx = K[0][0].item<float>();
    fy = K[1][1].item<float>();
    cx = K[0][2].item<float>();
    cy = K[1][2].item<float>();
    return res;
}

torch::Tensor Camera::loadDepth(float downscale_factor, float depth_scale)
{
    cv::Mat cDepth = cv::imread(depthFilePath, cv::IMREAD_UNCHANGED);

    // 将深度图像转换为浮点型
    cDepth.convertTo(cDepth, CV_32F);
    cDepth /= depth_scale;

    if (downscale_factor > 1.0f)
    {
        float scaleFactor = 1.0f / downscale_factor;
        cv::resize(cDepth, cDepth, cv::Size(), scaleFactor, scaleFactor, cv::INTER_AREA);
    }
    torch::Tensor res = depthToTensor(cDepth);
    return res;
}

std::string Camera::getFrameID(std::string prefix, std::string suffix) const
{
    // 找到前缀的位置
    size_t startPos = imgFilePath.find(prefix);
    if (startPos != std::string::npos)
    {
        // 移动到前缀之后的位置
        startPos += prefix.length();
        // 找到后缀的位置
        size_t endPos = imgFilePath.find(suffix, startPos);
        if (endPos != std::string::npos)
        {
            // 提取出所需的子字符串
            std::string frameNumber = imgFilePath.substr(startPos, endPos - startPos);
            return frameNumber;
        }
        else
        {
            std::cerr << "suffix invalid!" << std::endl;
        }
    }
    else
    {
        std::cerr << "prefix invalid!" << std::endl;
    }
    return "error";
}

// TODO: rename variables
TensorDict Camera::getFrameMaps() const
{
    torch::Tensor depth_map = depth.to(torch::kCUDA);
    torch::Tensor color_map = image.to(torch::kCUDA);
    torch::Tensor intrinsics = K.to(torch::kCUDA);
    torch::Tensor extrinsics = c2w_slam.to(torch::kCUDA);

    auto options = torch::TensorOptions().device(depth_map.device()).dtype(torch::kFloat32);

    int H = depth_map.size(0);
    int W = depth_map.size(1);

    auto cam_coords = computeVertexMap(depth_map, intrinsics);

    // 1. 重塑顶点图为 (H*W, 3)
    auto vertices_c = cam_coords.reshape({-1, 3});

    // 2. 将顶点转换为齐次坐标，添加第四个分量 1
    auto ones = torch::ones({vertices_c.size(0), 1}, vertices_c.options());
    auto vertices_c_homogeneous = torch::cat({vertices_c, ones}, 1);

    // 3. 应用外参矩阵
    auto vertices_w_homogeneous = torch::matmul(vertices_c_homogeneous, extrinsics.t());

    // 4. 转换回非齐次坐标
    auto vertices_world = vertices_w_homogeneous.slice(1, 0, 3) /
                          vertices_w_homogeneous.slice(1, 3, 4);

    // 5. 重塑回原始形状
    auto vertex_map = vertices_world.reshape({H, W, 3});

    auto normal_map = computeNormalMap(vertex_map);

    // // 计算法向图
    // auto normal_map = torch::zeros({height, width, 3}, options);

    // for (int y = 1; y < height - 1; ++y)
    // {
    //     for (int x = 1; x < width - 1; ++x)
    //     {
    //         auto dzdx = (vertex_map[y][x + 1][2] - vertex_map[y][x - 1][2]) / 2;
    //         auto dzdy = (vertex_map[y + 1][x][2] - vertex_map[y - 1][x][2]) / 2;
    //         auto normal = torch::stack({-dzdx, -dzdy, torch::ones({1}, options)}, 0);
    //         normal = normal / normal.norm();
    //         normal_map[y][x] = normal;
    //     }
    // }
    TensorDict frame_maps;
    frame_maps["color_map"] = color_map;
    frame_maps["vertex_map"] = vertex_map;
    frame_maps["normal_map"] = normal_map;

    return frame_maps;
}
DatasetReader::DatasetReader(const YAML::Node &config)
{
    input_dir = config["input_dir"].as<std::string>();
    this->config = config;

    // get camera params
    fx = config["intrinsics"].as<std::vector<float>>()[0];
    fy = config["intrinsics"].as<std::vector<float>>()[1];
    cx = config["intrinsics"].as<std::vector<float>>()[2];
    cy = config["intrinsics"].as<std::vector<float>>()[3];
    width = config["image_shape"].as<std::vector<int>>()[0];
    height = config["image_shape"].as<std::vector<int>>()[1];
    std::string pcd_name = config["pcd_name"].as<std::string>();
    downscale_factor = config["downscale_factor"].as<float>();
    scene_scale = 1.1f * config["scene_scale"].as<float>();
    scene_centor = torch::zeros(3);
    fs::path dataRoot(input_dir);
    // load init points
    fs::path pointsPath = dataRoot / pcd_name;
    if (fs::exists(pointsPath))
    {
        // 在初始点云文件存在的时候，读取初始点云
        std::cout << "read init scene point cloud." << std::endl;
        scene_points.readPly(pointsPath.string());
    }
    else
    {
        std::cout << "no init scene point cloud." << std::endl;
    }

    // check depth valid
    fs::path depthPath = dataRoot / config["depth_path"].as<std::string>();
    if (fs::exists(depthPath))
    {
        std::cout << "load data with depth!" << std::endl;
        has_depth = true;
    }
    else
    {
        std::cout << "load data without depth!" << std::endl;
        has_depth = false;
    }
}

std::vector<Camera> DatasetReader::getAllCams()
{
    auto all_cams = train_vec;
    if (config["test_split_interval"].as<int>() > 0)
        all_cams.insert(all_cams.end(), val_vec.begin(), val_vec.end());
    return all_cams;
}

void DatasetReader::read()
{
    // ProgressBar
    // Hide cursor
    show_console_cursor(false);
    ProgressBar bar{
        option::BarWidth{50},
        option::Start{"["},
        option::Fill{"="},
        option::Lead{">"},
        option::Remainder{" "},
        option::End{"]"},
        option::PostfixText{"Read cameras"},
        option::ForegroundColor{Color::green},
        option::ShowPercentage{true},
        option::ShowElapsedTime{true},
        option::ShowRemainingTime{true},
        option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}};

    // load path
    fs::path dataRoot(input_dir);
    fs::path imagePath = dataRoot / config["image_path"].as<std::string>();
    fs::path posePath = dataRoot / config["pose_path"].as<std::string>();
    fs::path depthPath = dataRoot / config["depth_path"].as<std::string>();
    // load id
    int start_frame = config["start_frame"].as<int>();
    int end_frame = config["end_frame"].as<int>();
    int frame_step = config["frame_step"].as<int>();
    int test_split_interval = config["test_split_interval"].as<int>();

    if (end_frame <= 0)
    {
        int jpg_count = 0;
        for (const auto &entry : fs::directory_iterator(imagePath))
        {
            if (entry.is_regular_file() && entry.path().extension() == ".jpg")
            {
                jpg_count++;
            }
        }
        end_frame = jpg_count;
    }

    int read_count = 0;
    int total_frame = (end_frame - start_frame + 1) / frame_step;
    torch::Tensor init_c2w;
    for (int frame_id = start_frame; frame_id <= end_frame; frame_id += frame_step)
    {

        std::string filename = idToFilename(frame_id);
        std::string pose_filename = (posePath / ("pose" + filename + ".txt")).string();
        std::string img_filename = (imagePath / ("frame" + filename + ".jpg")).string();
        std::string depth_filename = (depthPath / ("depth" + filename + ".png")).string();
        if (!fs::exists(pose_filename))
            throw std::runtime_error(pose_filename + " does not exist");
        if (!fs::exists(img_filename))
            throw std::runtime_error(img_filename + " does not exist");
        if (has_depth && !fs::exists(depth_filename))
            throw std::runtime_error(depth_filename + " does not exist");
        Matrix4 c2w_mat = readMatrixFromTXT(pose_filename);
        torch::Tensor c2w_tensor = matrixToTensor(c2w_mat);
        torch::Tensor ref_c2w_pose;
        if (read_count == 0)
        {
            init_c2w = c2w_tensor;
        }
        ref_c2w_pose = torch::matmul(poseInv(init_c2w), c2w_tensor);
        Camera cam(width, height, fx, fy, cx, cy, has_depth, ref_c2w_pose, img_filename, depth_filename);

        cam.c2w_slam = ref_c2w_pose.clone();
        cam.image = cam.loadImage(downscale_factor);
        cam.id = read_count;
        if (has_depth)
            cam.depth = cam.loadDepth(downscale_factor, config["depth_scale"].as<float>());

        // cam.toGPU();

        if (test_split_interval < 0)
        {
            // 不区分训练、测试数据集
            train_vec.push_back(cam);
            val_vec.push_back(cam);
        }
        else
        {
            // 按照比例区分训练/测试数据集
            if (read_count % test_split_interval == 0)
                val_vec.push_back(cam);
            else
            {
                train_vec.push_back(cam);
            }
        }
        read_count++;
        bar.set_option(option::PostfixText{"Reading camera: " + std::to_string(read_count) + "/" + std::to_string(total_frame)});
        bar.set_progress(100 * read_count / total_frame);
    }
    show_console_cursor(true);
    std::cout << "train set num: " << train_vec.size() << std::endl;
    std::cout << "val set num: " << val_vec.size() << std::endl;
}

void DatasetReader::updateSceneGeo()
{
    if (train_vec.size() == 1)
    {
        // 当只有一个相机的时候，无法估算场景几何，直接返回
        scene_scale = 1.0f;
        scene_centor = torch::zeros({3});
        return;
    }
    torch::Tensor sum = torch::zeros({3});
    auto all_cams = getAllCams();

    for (auto cam : all_cams)
    {
        torch::Tensor cam_location = cam.c2w.index({Slice(0, 3), 3});
        sum += cam_location;
    }

    scene_centor = sum / train_vec.size();
    float max_dist = 0.0;
    for (auto cam : all_cams)
    {
        torch::Tensor cam_location = cam.c2w.index({Slice(0, 3), 3});
        float dist = torch::norm(cam_location - scene_centor).item<float>();
        if (dist > max_dist)
        {
            max_dist = dist;
        }
    }
    scene_scale = 1.1f * max_dist;
    std::cout << "scene centor: " << scene_centor[0].item<float>() << "\t" << scene_centor[1].item<float>() << "\t" << scene_centor[2].item<float>() << std::endl;
    std::cout << "scene scale: " << scene_scale << std::endl;
}

void DatasetReader::savePose(const std::string &save_dir)
{
    fs::create_directories(save_dir);
    // std::cout << save_dir << std::endl;
    for (size_t i = 0; i < train_vec.size(); i++)
    {

        torch::Tensor slam_pose = train_vec[i].c2w_slam;
        // std::cout << slam_pose << std::endl;
        std::string pose_file_name = "frame" + train_vec[i].getFrameID() + ".txt";
        std::string save_file_name = fs::path(save_dir) / pose_file_name;
        saveTensorTXT(slam_pose, save_file_name);
    }
}

void saveCameras(const std::vector<Camera> &input_cameras, const std::string &filename)
{
    json j = json::array();
    // std::cout << "save Camera num: " << input_cameras.size() << std::endl;
    for (size_t i = 0; i < input_cameras.size(); i++)
    {
        const Camera &cam = input_cameras[i];

        json camera = json::object();
        camera["id"] = i;
        camera["img_name"] = fs::path(cam.imgFilePath).filename().string();
        camera["width"] = cam.width;
        camera["height"] = cam.height;
        camera["fx"] = cam.fx;
        camera["fy"] = cam.fy;

        torch::Tensor R = cam.c2w_slam.index({Slice(None, 3), Slice(None, 3)});
        torch::Tensor T = cam.c2w_slam.index({Slice(None, 3), Slice(3, 4)}).squeeze();

        std::vector<float> position(3);
        std::vector<std::vector<float>> rotation(3, std::vector<float>(3));
        for (int i = 0; i < 3; i++)
        {
            position[i] = T[i].item<float>();
            for (int j = 0; j < 3; j++)
            {
                rotation[i][j] = R[i][j].item<float>();
            }
        }

        camera["position"] = position;
        camera["rotation"] = rotation;
        j.push_back(camera);
    }

    std::ofstream of(filename);
    of << j;
    of.close();

    std::cout << "Wrote camera json to: " << filename << std::endl;
}