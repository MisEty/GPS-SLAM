import numpy as np
import cv2
import os
import shutil
import re
from tqdm import tqdm
import glob


def generate_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    os.mkdir(path + "/camera")


def get_color_images(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    pattern = re.compile(r'^frame\d{6}\.jpg$')
    img_count = 0
    for filename in os.listdir(src_dir):
        if pattern.match(filename):
            src_path = os.path.join(src_dir, filename)
            dst_path = os.path.join(dst_dir, filename)

            shutil.copy2(src_path, dst_path)
            img_count += 1

    print("color finish, count: {}".format(img_count))

def get_depths(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    pattern = re.compile(r'^depth\d{6}\.png$')
    img_count = 0
    for filename in os.listdir(src_dir):
        if pattern.match(filename):
            src_path = os.path.join(src_dir, filename)
            dst_path = os.path.join(dst_dir, filename)

            shutil.copy2(src_path, dst_path)
            img_count += 1

    print("color finish, count: {}".format(img_count))

def get_intrinsics(fx, fy, cx, cy):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def get_color_extrinsics(traj_path, save_dir):
    raw_traj = np.loadtxt(traj_path)
    raw_traj = raw_traj.reshape((raw_traj.shape[0], 4, 4))
    pose_count = 0
    for i, matrix in enumerate(raw_traj):
        filename = save_dir + f"/pose{str(i).zfill(6)}.txt"
        np.savetxt(filename, matrix, fmt='%.8f')
        pose_count += 1
    print("pose finish, count: {}".format(pose_count))
    return raw_traj


def sample_and_rename_poses(input_dir, sample_interval=10):
    pose_files = sorted(glob.glob(os.path.join(input_dir, "pose*.txt")))
    total_poses = len(pose_files)
    
    sampled_count = (total_poses + sample_interval - 1) // sample_interval
    
    print(f"Total poses: {total_poses}")
    print(f"After sampling: {sampled_count}")
    
    temp_dir = os.path.join(input_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    new_idx = 0
    for i in tqdm(range(0, total_poses, sample_interval)):
        if i >= total_poses:
            break
            
        src_file = pose_files[i]
        dst_file = os.path.join(temp_dir, f"pose{new_idx:06d}.txt")
        
        with open(src_file, 'r') as f_src:
            content = f_src.read()
        with open(dst_file, 'w') as f_dst:
            f_dst.write(content)
            
        new_idx += 1
    
    for pose_file in pose_files:
        os.remove(pose_file)
    
    sampled_files = glob.glob(os.path.join(temp_dir, "pose*.txt"))
    for sampled_file in sampled_files:
        filename = os.path.basename(sampled_file)
        os.rename(sampled_file, os.path.join(input_dir, filename))
    
    os.rmdir(temp_dir)
    
    print("Sampling and renaming completed!")


def sample_and_rename_frames(input_dir, sample_interval=10):
    frame_files = sorted(glob.glob(os.path.join(input_dir, "frame*.jpg")))
    total_frames = len(frame_files)
    
    sampled_count = (total_frames + sample_interval - 1) // sample_interval
    
    print(f"Total frames: {total_frames}")
    print(f"After sampling: {sampled_count}")
    
    temp_dir = os.path.join(input_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    new_idx = 0
    for i in tqdm(range(0, total_frames, sample_interval)):
        if i >= total_frames:
            break
            
        src_file = frame_files[i]
        dst_file = os.path.join(temp_dir, f"frame{new_idx:06d}.jpg")
        
        shutil.copy2(src_file, dst_file)
        new_idx += 1
    
    for frame_file in frame_files:
        os.remove(frame_file)
    
    sampled_files = glob.glob(os.path.join(temp_dir, "frame*.jpg"))
    for sampled_file in sampled_files:
        filename = os.path.basename(sampled_file)
        os.rename(sampled_file, os.path.join(input_dir, filename))
    
    os.rmdir(temp_dir)
    print("Sampling and renaming completed!")

def sample_and_rename_depths(input_dir, sample_interval=10):
    frame_files = sorted(glob.glob(os.path.join(input_dir, "depth*.png")))
    total_frames = len(frame_files)
    
    sampled_count = (total_frames + sample_interval - 1) // sample_interval
    
    print(f"Total frames: {total_frames}")
    print(f"After sampling: {sampled_count}")
    
    temp_dir = os.path.join(input_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    new_idx = 0
    for i in tqdm(range(0, total_frames, sample_interval)):
        if i >= total_frames:
            break
            
        src_file = frame_files[i]
        dst_file = os.path.join(temp_dir, f"depth{new_idx:06d}.png")
        
        shutil.copy2(src_file, dst_file)
        new_idx += 1
    
    for frame_file in frame_files:
        os.remove(frame_file)
    
    sampled_files = glob.glob(os.path.join(temp_dir, "depth*.png"))
    for sampled_file in sampled_files:
        filename = os.path.basename(sampled_file)
        os.rename(sampled_file, os.path.join(input_dir, filename))
    
    os.rmdir(temp_dir)
    
    print("Sampling and renaming completed!")


# same for all replica dataset!
fx = 600
fy = 600
cx = 599.5
cy = 339.5
w = 1200
h = 680
scale = 6553.5

input_dir = "data/Replica_raw/office1" # raw format replica scene dir
output_dir = "data/replica/office1" # ours format replica scene dir
frame_sample_num = 2000

generate_dir(output_dir)
color_poses = get_color_extrinsics(input_dir + "/traj.txt", output_dir + "/camera")
frame_num = color_poses.shape[0]
get_color_images((input_dir + "/results"), output_dir + "/camera")
get_depths((input_dir + "/results"), output_dir + "/depth")
intrinsics = get_intrinsics(fx, fy, cx, cy)
np.savetxt(output_dir + "/camera/intrinsics.txt", intrinsics, fmt='%.8f')
img_shape = np.array([w, h]).astype(np.int32)
np.savetxt(output_dir + "/camera/img_shape.txt", img_shape, fmt='%d')


if (frame_num != frame_sample_num):
    sample_and_rename_poses(os.path.join(output_dir,"camera"),int(frame_num / frame_sample_num))
    sample_and_rename_frames(os.path.join(output_dir,"camera"),int(frame_num / frame_sample_num))
    sample_and_rename_depths(os.path.join(output_dir,"depth"),int(frame_num / frame_sample_num))
