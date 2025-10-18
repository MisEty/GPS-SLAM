#ifndef DATASET_READER_H
#define DATASET_READER_H

#include <filesystem>
#include <vector>
#include <random>
#include <algorithm>

#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <indicators/progress_bar.hpp>
#include <indicators/cursor_control.hpp>

#include "yaml-cpp/yaml.h"

using namespace indicators;
typedef std::map<std::string, torch::Tensor> TensorDict;

template <typename T>
struct IndexedElement
{
    T value;
    size_t originalIndex;
};

template <typename T>
class RandomSelector
{
private:
    std::vector<IndexedElement<T>> original;
    std::vector<IndexedElement<T>> current;
    std::vector<double> probabilities;
    std::mt19937 gen;
    std::uniform_real_distribution<> dis;
    bool useProbabilities;

public:
    RandomSelector(const std::vector<T> &vec, const std::vector<double> &probs = {})
        : gen(std::random_device{}()), dis(0.0, 1.0)
    {
        original.reserve(vec.size());
        for (size_t i = 0; i < vec.size(); ++i)
        {
            original.push_back({vec[i], i});
        }
        current = original;

        if (probs.empty())
        {
            useProbabilities = false;
        }
        else
        {
            useProbabilities = true;
            probabilities = probs;
            // 归一化概率
            double sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
            for (auto &prob : probabilities)
            {
                prob /= sum;
            }
        }
    }

    IndexedElement<T> getNext()
    {
        if (!useProbabilities)
        {
            if (current.empty())
            {
                current = original;
            }

            std::uniform_int_distribution<> intDis(0, current.size() - 1);
            int index = intDis(gen);
            IndexedElement<T> value = std::move(current[index]);

            std::swap(current[index], current.back());
            current.pop_back();

            return value;
        }
        else
        {
            double randomValue = dis(gen);
            double cumulativeProbability = 0.0;

            for (size_t i = 0; i < probabilities.size(); ++i)
            {
                cumulativeProbability += probabilities[i];
                if (randomValue <= cumulativeProbability)
                {
                    return original[i];
                }
            }
            // 如果由于浮点数精度问题没有选中，则返回第一个元素
            return original.front();
        }
    }
};

struct Points
{
    torch::Tensor xyz;
    torch::Tensor rgb;
    torch::Tensor normal;

    void readPly(std::string filename);
};

struct Camera
{
    int id = -1;
    int width = 0;
    int height = 0;
    float fx = 0;
    float fy = 0;
    float cx = 0;
    float cy = 0;
    bool has_depth;
    torch::Tensor image;
    torch::Tensor c2w;      // camera to world transform (gt)
    torch::Tensor c2w_slam; // camera to world transform (slam estiamted)
    torch::Tensor depth;

    torch::Tensor K;
    std::string imgFilePath = "";
    std::string depthFilePath = "";

    Camera() {};

    Camera(int width, int height, float fx, float fy, float cx, float cy, bool has_depth, const torch::Tensor &c2w, const std::string &imgFilePath = "", const std::string &depthFilePath = "") : width(width), height(height), fx(fx), fy(fy), cx(cx), cy(cy), has_depth(has_depth), c2w(c2w), imgFilePath(imgFilePath), depthFilePath(depthFilePath)
    {
        c2w_slam = c2w;
        K = torch::tensor({{fx, 0.0f, cx},
                           {0.0f, fy, cy},
                           {0.0f, 0.0f, 1.0f}},
                          torch::kFloat32);
    }

    std::string getFrameID(std::string prefix = "frame",
                           std::string suffix = ".jpg") const;

    torch::Tensor loadImage(float downscale_factor);

    torch::Tensor loadDepth(float downscale_factor, float depth_scale);

    TensorDict getFrameMaps() const;

    void toGPU()
    {
        torch::Device device = torch::kCUDA;
        image = image.to(device);
        c2w = c2w.to(device);
        c2w_slam = c2w_slam.to(device);
        depth = depth.to(device);
        K = K.to(device);
    }

    void toCPU()
    {
        torch::Device device = torch::kCPU;
        image = image.to(device);
        c2w = c2w.to(device);
        c2w_slam = c2w_slam.to(device);
        depth = depth.to(device);
        K = K.to(device);
    }
};

void saveCameras(const std::vector<Camera> &input_cameras, const std::string &filename);

class DatasetReader
{
public:
    YAML::Node config;
    // 初始点云
    Points scene_points;
    torch::Tensor scene_centor;
    float scene_scale;
    // 相机参数
    float fx, fy, cx, cy;
    int width, height;
    float downscale_factor;
    std::string input_dir;
    // 相机数据序列
    std::vector<Camera> train_vec = std::vector<Camera>();
    std::vector<Camera> val_vec = std::vector<Camera>();
    // 是否包含深度图像
    bool has_depth;

    DatasetReader() = default;

    ~DatasetReader() = default;

    DatasetReader(const YAML::Node &config);

    void read();

    void updateSceneGeo();

    std::vector<Camera> getAllCams();

    void savePose(const std::string &save_dir);
};

#endif // DATASET_READER_H