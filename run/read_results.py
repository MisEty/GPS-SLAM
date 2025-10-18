import os
import re
import argparse
import json
import csv  # 新增导入用于处理CSV文件


def process_file(file_path):
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        per_frame_time = None
        fps = None
        gpu_memory = None

        for line in lines:
            line = line.strip()
            if line.startswith("[PIPELINE AVG TIME]"):
                # 提取 per frame fusion time
                match = re.search(r"per frame fusion time: (\d+\.\d+)", line)
                if match:
                    per_frame_time = float(match.group(1))
                # 提取 FPS
                match_fps = re.search(r"FPS: (\d+\.\d+)", line)
                if match_fps:
                    fps = float(match_fps.group(1))
            elif line.startswith("GPU memory usage:"):
                match = re.search(r"GPU memory usage: (\d+) MB", line)
                if match:
                    gpu_memory = int(match.group(1))

        if per_frame_time is None or fps is None or gpu_memory is None:
            print(f"Warning: Skipping {file_path} due to missing data.")
            return None  # 返回None表示数据不完整

        # 计算FPS指标
        calculated_fps = 1000 / per_frame_time
        gaussian_fps = 1000 / (1000 / fps - per_frame_time)
        file_dir = os.path.dirname(file_path)

        # 处理JSON文件
        json_path = os.path.join(file_dir, "val", "results.json")
        ssim = None
        psnr = None
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as jf:
                    json_data = json.load(jf)
                ssim = json_data.get("SSIM")  # 使用get方法避免KeyError
                psnr = json_data.get("PSNR")
                lpips = json_data.get("LPIPS")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in {json_path}: {e}")
            except Exception as e:
                print(f"Error reading {json_path}: {e}")
        else:
            print(f"Note: {json_path} does not exist.")
        res = {
            "RootDir": file_dir,
            "Fusion_FPS": round(calculated_fps, 2),
            "Gaussian_FPS": round(gaussian_fps, 2),
            "FPS": round(fps, 2),
            "GPU_Memory": gpu_memory,
            "SSIM": ssim,
            "PSNR": psnr,
            "LPIPS": lpips,
        }
        print(res)
        # 返回结构化数据字典
        return res
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Process timelog files")
    parser.add_argument(
        "--root_dir", type=str, help="Root directory to scan for timelog.txt files"
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help="If provided, only process files with paths containing this string.",
    )
    args = parser.parse_args()

    results = []  # 存储所有有效数据

    # 遍历目录树
    for root, _, files in os.walk(args.root_dir):
        for file in files:
            if file == "time_log.txt":
                file_path = os.path.join(root, file)
                # 应用过滤器
                if args.filter and args.filter not in file_path:
                    continue  # 跳过不匹配的文件
                data = process_file(file_path)
                if data:
                    results.append(data)  # 收集有效数据

    # 写入CSV文件
    if results:
        csv_path = os.path.join(args.root_dir, "results.csv")
        fieldnames = [
            "RootDir",
            "Fusion_FPS",
            "Gaussian_FPS",
            "FPS",
            "GPU_Memory",
            "SSIM",
            "PSNR",
            "LPIPS",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to {csv_path}")
        means = {key: 0 for key in fieldnames if key != "RootDir"}
        for result in results:
            for key in means:
                means[key] += result[key] if result[key] is not None else 0
        means = {key: value / len(results) for key, value in means.items()}
        print("Means:", means)
    else:
        print("No valid data to save.")


if __name__ == "__main__":
    main()
