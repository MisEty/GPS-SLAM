#include "dataset_reader.h"
#include "slam_pipeline.h"
#include "InfiniTAM_tools.h"
#include <boost/asio.hpp>
#include "json.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;
using namespace torch::indexing;
using namespace torch::autograd;

Camera readMessage(boost::asio::ip::tcp::socket &sock)
{
    char lengthBuffer[4];
    boost::asio::read(sock, boost::asio::buffer(lengthBuffer, 4));
    int messageLength = *reinterpret_cast<int *>(lengthBuffer);
    // 读取消息内容
    std::vector<char> messageBuffer(messageLength);
    boost::asio::read(sock, boost::asio::buffer(messageBuffer, messageLength));
    // 解析 JSON 消息
    std::string messageStr(messageBuffer.begin(), messageBuffer.end());
    json message = json::parse(messageStr);
    // std::cout << message << std::endl;
    float fov_x = message["fov_x"];
    float fov_y = message["fov_y"];
    float resolution_x = message["resolution_x"];
    float resolution_y = message["resolution_y"];
    float fx = resolution_x / (2.0f * tan(fov_x / 2.0f));
    float fy = resolution_y / (2.0f * tan(fov_y / 2.0f));
    float cx = resolution_x / 2;
    float cy = resolution_y / 2;
    std::vector<float> pose_matrix_data = message["pose"];
    // client的eigen mat默认列优先，而libtorch tensor默认行优先，读取后需要转置
    torch::Tensor pose_matrix = torch::from_blob(pose_matrix_data.data(), {4, 4}, torch::TensorOptions().dtype(torch::kFloat32)).transpose(0, 1);
    pose_matrix.index({torch::indexing::Slice(), 1}) *= -1;
    pose_matrix.index({torch::indexing::Slice(), 2}) *= -1;
    Camera cam(int(resolution_x), int(resolution_y), fx, fy, cx, cy, false, pose_matrix.clone());
    return cam;
}

void sendImage(boost::asio::ip::tcp::socket &sock, const cv::Mat &img)
{
    cv::Mat send_img;
    uint32_t img_width = img.cols;
    uint32_t img_height = img.rows;
    boost::asio::write(sock, boost::asio::buffer(&img_width, sizeof(uint32_t)));
    boost::asio::write(sock, boost::asio::buffer(&img_height, sizeof(uint32_t)));
    uint32_t img_byte = img_width * img_height * 3;
    cv::cvtColor(img, send_img, cv::COLOR_BGR2RGB);
    boost::asio::write(sock, boost::asio::buffer(send_img.data, img_byte));
}

void sendTensor(boost::asio::ip::tcp::socket &sock, const torch::Tensor &tensor)
{
    torch::Tensor contiguous_tensor = tensor.contiguous();
    float *data_ptr = contiguous_tensor.data_ptr<float>();
    uint32_t num_bytes = contiguous_tensor.numel() * sizeof(float);
    boost::asio::write(sock, boost::asio::buffer(data_ptr, num_bytes));
}

void sendString(boost::asio::ip::tcp::socket &sock, const std::string &message)
{
    uint32_t infoLen = static_cast<uint32_t>(message.size());
    boost::asio::write(sock, boost::asio::buffer(&infoLen, sizeof(uint32_t)));
    boost::asio::write(sock, boost::asio::buffer(message.data(), infoLen));
}

int main(int argc, char *argv[])
{
    std::cout << "ours viewer server!" << std::endl;
    const char *config_filename = argv[1];
    YAML::Node config = YAML::LoadFile(config_filename);
    std::string workspace_dir = config["workspace_dir"].as<std::string>();
    std::string work_mode = config["work_mode"].as<std::string>();

    // setup cout precision
    std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);

    // setup cuda device
    const std::string devId = config["dev_id"].as<std::string>();
    setenv("CUDA_VISIBLE_DEVICES", devId.c_str(), 1);
    DatasetReader data_reader(config["READER"]);
    data_reader.read();
    data_reader.updateSceneGeo();
    CLIEngine *tsdf_engine = createTsdfEngine(data_reader, config["PIPE"]["TSDF"]);
    SLAMPipeline pipe;
    pipe.setTsdfEngine(tsdf_engine);
    pipe.work_mode = work_mode;
    pipe.device_id = config["dev_id"].as<int>();

    std::cout << "======= setup Gaussian model ======" << std::endl;
    SLAMGaussianModel model;
    model.loadConfig(config["MODEL"]);
    model.loadParamsTensor(workspace_dir + "/gs_model/model.pt");
    pipe.scene_scale = data_reader.scene_scale;
    pipe.loadConfig(config["PIPE"], workspace_dir, false);
    pipe.loadEngine();

    std::cout << "======= start viewer as server ======" << std::endl;
    bool keep_running = true;
    int _port = config["port"].as<int>();
    torch::Device device = torch::kCUDA;
    float depth_vis_max = pipe.vis_configs["depth_vis_max"].as<float>();
    try
    {
        // io_service对象
        boost::asio::io_service ios;
        // 绑定端口6688
        boost::asio::ip::tcp::acceptor acceptor(ios, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), _port));
        boost::asio::ip::tcp::socket sock(ios);
        // 阻塞等待socket连接
        acceptor.accept(sock);

        std::cout << "client connected!" << std::endl;
        int time_stamp = 0;
        while (keep_running)
        {
            // 通过json文件进行通讯, 接收数据，创建相机
            Camera cam = readMessage(sock);
            time_stamp++;
            TensorDict raycast_res = pipe.runRaycastByCam(cam, false);
            torch::Tensor raycast_color = raycast_res["color_map"];
            torch::Tensor raycast_depth = raycast_res["depth_map"];
            TensorDict render_res = model.forward(cam, raycast_depth, raycast_color);
            torch::Tensor rendered_rgb = torch::clamp(render_res["rgb"], 0, 1);
            cv::Mat raycast_color_img = tensorToImage(raycast_color);
            cv::Mat raycast_depth_img = tensorToJetMat(raycast_depth, 0, depth_vis_max, true);

            cv::Mat rendered_color_img = tensorToImage(rendered_rgb);
            sendImage(sock, rendered_color_img);
            // send input color
            cv::Mat input_color_img = rendered_color_img.clone();
            sendImage(sock, input_color_img);
            // send raycast color
            sendImage(sock, raycast_color_img);
            // send raycast depth
            sendImage(sock, raycast_depth_img);
            // send current pose, info and mvp matrix
            torch::Tensor curr_pose = cam.c2w_slam;
            auto rot = curr_pose.index({Slice(0, 3), Slice(0, 3)});
            auto trans = curr_pose.index({Slice(None, 3), Slice(3, 4)});
            sendTensor(sock, rot);
            sendTensor(sock, trans);
            // send string info
            std::string info = "debug test";
            sendString(sock, info);
            // send mvp
            torch::Tensor mvp = curr_pose;
            sendTensor(sock, mvp);
        }
    }
    catch (std::exception &e)
    {
        std::cout << "exception: " << e.what() << std::endl;
    }
}
