#include "file_utils.h"

namespace fs = std::filesystem;

///////////////// EIGEN LOAD AND SAVE /////////////////

MatrixX readMatrixFromTXT(const std::string &filename)
{
    std::ifstream infile(filename);
    std::vector<std::vector<double>> matrix;

    if (!infile.is_open())
    {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return Eigen::MatrixXf();
    }

    std::string line;
    while (std::getline(infile, line))
    {
        std::vector<double> row;
        std::stringstream ss(line);
        double value;

        while (ss >> value)
        {
            row.push_back(value);
        }

        if (!row.empty())
        {
            matrix.push_back(row);
        }
    }

    infile.close();

    if (matrix.empty())
    {
        std::cerr << "Error: File is empty or contains invalid data." << std::endl;
        return Eigen::MatrixXf();
    }

    int rows = matrix.size();
    int cols = matrix[0].size();

    Eigen::MatrixXf eigenMatrix(rows, cols);

    for (int i = 0; i < rows; ++i)
    {
        if (static_cast<int>(matrix[i].size()) != cols)
        {
            std::cerr << "Error: Inconsistent number of columns in the file." << std::endl;
            return Eigen::MatrixXf();
        }
        for (int j = 0; j < cols; ++j)
        {
            eigenMatrix(i, j) = matrix[i][j];
        }
    }

    return eigenMatrix;
}

torch::Tensor matrixToTensor(const MatrixX &eigenMatrix)
{
    // 获取 Eigen 矩阵的尺寸
    int rows = eigenMatrix.rows();
    int cols = eigenMatrix.cols();
    // 创建一个 torch::Tensor，使用 Eigen 矩阵的数据
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor tensor;
    if (eigenMatrix.IsRowMajor)
    {
        // 如果 Eigen 矩阵是行优先的，直接创建 tensor
        tensor = torch::from_blob(
                     const_cast<float *>(eigenMatrix.data()),
                     {rows, cols},
                     options)
                     .clone();
    }
    else
    {
        // 如果 Eigen 矩阵是列优先的，创建 tensor 后进行转置
        tensor = torch::from_blob(
                     const_cast<float *>(eigenMatrix.data()),
                     {cols, rows},
                     options)
                     .clone()
                     .transpose(0, 1);
    }
    return tensor;
}

////////////////////////////////////////////////////////////////

bool createDirectory(const std::string &path, bool overwrite = false)
{
    try
    {
        fs::path dirPath(path);

        // Check if the path already exists
        if (fs::exists(dirPath))
        {
            if (overwrite)
            {
                // If overwrite is true, remove the existing directory and all its contents
                fs::remove_all(dirPath);
            }
            else
            {
                // If overwrite is false and the directory exists, return false
                std::cout << "Directory already exists, no action taken: " << path << std::endl;
                return false;
            }
        }

        // Create the directory
        if (fs::create_directories(dirPath))
        {
            std::cout << "Directory created successfully: " << path << std::endl;
            return true;
        }
        else
        {
            std::cout << "Failed to create directory: " << path << std::endl;
            return false;
        }
    }
    catch (const fs::filesystem_error &e)
    {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
        return false;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        return false;
    }
}

std::string idToFilename(int frame_id)
{
    std::ostringstream filename;
    filename << std::setw(6) << std::setfill('0') << frame_id;
    return filename.str();
}

void createWorkSpace(const std::string &config_filename)
{
    YAML::Node config = YAML::LoadFile(config_filename);
    auto workspace_dir = config["workspace_dir"].as<std::string>();
    createDirectory(workspace_dir, true);
    // copy config file
    fs::path source = fs::path(config_filename);
    fs::path target_dir = config["workspace_dir"].as<std::string>();
    fs::path target = target_dir / source.filename();
    fs::copy_file(source, target, fs::copy_options::overwrite_existing);
}

void saveCfgArgs(const std::string &file_name, int maxSH)
{
    std::ofstream outfile(file_name);
    outfile
        << "Namespace(data_device='cuda', eval=False, images='images', model_path='', resolution=-1, sh_degree=" << maxSH << ", source_path='', white_background=False)"
        << std::endl;
    outfile.close();
}

// Function to get the memory usage of the GPU
unsigned long long getGPUMemoryUsage(int gpu_id)
{
    nvmlReturn_t result;
    nvmlDevice_t device;
    nvmlMemory_t memory;

    // Initialize NVML library
    result = nvmlInit();
    if (NVML_SUCCESS != result)
    {
        std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result) << "\n";
        return ULLONG_MAX;
    }

    // Get the GPU handle
    result = nvmlDeviceGetHandleByIndex(gpu_id, &device);
    if (NVML_SUCCESS != result)
    {
        std::cerr << "Failed to get handle for device " << std::to_string(gpu_id) << ": " << nvmlErrorString(result) << "\n";
        nvmlShutdown();
        return ULLONG_MAX;
    }

    // Get memory information
    result = nvmlDeviceGetMemoryInfo(device, &memory);
    if (NVML_SUCCESS != result)
    {
        std::cerr << "Failed to get memory info: " << nvmlErrorString(result) << "\n";
        nvmlShutdown();
        return ULLONG_MAX;
    }

    auto used_gpu_memory = memory.used / (1024 * 1024);

    // Shutdown NVML library
    nvmlShutdown();

    return used_gpu_memory;
}

double calculateTimeInterval(struct timespec start, struct timespec end)
{
    // 计算秒和纳秒的时间差
    time_t seconds_diff = end.tv_sec - start.tv_sec;
    long nanoseconds_diff = end.tv_nsec - start.tv_nsec;

    // 将时间差转换为毫秒
    double interval_ms = (seconds_diff * 1000.0) + (nanoseconds_diff / 1000000.0);

    return interval_ms;
}

void saveTensorTXT(const torch::Tensor &tensor, const std::string &filename)
{
// 检查输入是否为二维张量
    if (tensor.dim() != 2 || tensor.size(0) != 4 || tensor.size(1) != 4) {
        throw std::invalid_argument("输入必须是4x4的二维张量");
    }

    // 确保张量内存连续（防止非连续内存导致访问错误）
    auto tensor_contig = tensor.contiguous();

    // 打开文件流
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件: " + filename);
    }

    // 设置输出流的浮点精度为固定6位小数
    file << std::fixed << std::setprecision(6);

    // 获取数据指针（假设数据类型为float）
    auto data_ptr = tensor_contig.data_ptr<float>();

    // 按行写入数据
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            // 计算一维索引并写入值
            file << data_ptr[row * 4 + col];
            if (col < 3) file << " "; // 列间加空格
        }
        file << "\n"; // 每行结束后换行
    }

    file.close();
}