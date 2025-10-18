python run/train.py --config-dir configs/release
python run/eval.py --base-path output/release
python run/read_results.py --root_dir output/release/replica
python run/read_results.py --root_dir output/release/gps_slam