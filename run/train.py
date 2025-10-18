import os
import subprocess
import argparse


def run_config(executable, config_path):
    command = [executable, config_path]
    print(f"Running: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
        print(f"Finished running {config_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {config_path}: {e}")
    print("-" * 50)


def process_configs(executable, folder):
    for root, dirs, files in os.walk(folder):
        for file in sorted(files):
            if file.endswith(".yaml"):
                config_path = os.path.join(root, file)
                run_config(executable, config_path)


def main():
    parser = argparse.ArgumentParser(description="Process YAML configurations.")
    parser.add_argument(
        "--executable",
        type=str,
        default="./build/slam_trainer",
        help="Path to the executable (default: ./build/slam_trainer)",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        required=True,
        help="Root folder of configuration files",
    )
    args = parser.parse_args()

    executable = args.executable
    config_root_folder = args.config_dir

    print(
        f"Starting to process all YAML files in {config_root_folder} and its subfolders."
    )
    process_configs(executable, config_root_folder)
    print("All configurations have been processed.")


if __name__ == "__main__":
    main()
