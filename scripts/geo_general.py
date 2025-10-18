import open3d as o3d
import trimesh
import numpy as np
from scipy.spatial import cKDTree as KDTree
from tqdm import tqdm
from argparse import ArgumentParser


def completion_ratio(gt_points, rec_points, dist_th=0.03):
    gen_points_kd_tree = KDTree(rec_points)
    distances, _ = gen_points_kd_tree.query(gt_points)
    comp_ratio = np.mean((distances < dist_th).astype(np.float32))
    return comp_ratio


def accuracy_ratio(gt_points, rec_points, dist_th=0.03):
    gt_points_kd_tree = KDTree(gt_points)
    distances, _ = gt_points_kd_tree.query(rec_points)
    acc_ratio = np.mean((distances < dist_th).astype(np.float32))
    return acc_ratio


def accuracy(gt_points, rec_points):
    gt_points_kd_tree = KDTree(gt_points)
    distances, _ = gt_points_kd_tree.query(rec_points)
    acc = np.mean(distances)
    return acc


def completion(gt_points, rec_points):
    gt_points_kd_tree = KDTree(rec_points)
    distances, _ = gt_points_kd_tree.query(gt_points)
    comp = np.mean(distances)
    return comp


def eval_pcd(
    rec_meshfile,
    gt_meshfile,
    transform=np.eye(4),
    dist_thres=[0.03],
    sample_nums=1000000,
):
    """
    3D reconstruction metric.

    """
    mesh_gt = trimesh.load(gt_meshfile, process=False)
    bbox = np.zeros([2, 3])
    bbox[0] = mesh_gt.vertices.min(axis=0) - 0.05
    bbox[1] = mesh_gt.vertices.max(axis=0) + 0.05
    rec_pc = o3d.io.read_point_cloud(rec_meshfile)
    rec_pc.transform(transform)

    points = np.asarray(rec_pc.points)
    P = points.shape[0]
    print("recon points num:", P)
    points = points[np.random.choice(P, min(P, sample_nums), replace=False), :]
    rec_pc_tri = trimesh.PointCloud(vertices=points)

    gt_pc = trimesh.sample.sample_surface(mesh_gt, sample_nums)
    gt_pc_tri = trimesh.PointCloud(vertices=gt_pc[0])
    print("compute acc")
    accuracy_rec = accuracy(gt_pc_tri.vertices, rec_pc_tri.vertices)
    print("compute comp")
    completion_rec = completion(gt_pc_tri.vertices, rec_pc_tri.vertices)
    Ps = {}
    Rs = {}
    Fs = {}
    for thre in tqdm(dist_thres):
        P = accuracy_ratio(gt_pc_tri.vertices, rec_pc_tri.vertices, dist_th=thre) * 100
        R = (
            completion_ratio(gt_pc_tri.vertices, rec_pc_tri.vertices, dist_th=thre)
            * 100
        )
        F1 = 2 * P * R / (P + R)
        Ps["accuracy ratio (< {})".format(thre)] = P
        Rs["completion ratio (< {})".format(thre)] = R
        Fs["F1 (< {})".format(thre)] = F1
    accuracy_rec *= 100  # convert to cm
    completion_rec *= 100  # convert to cm
    results = {
        "accuracy": accuracy_rec,
        "completion": completion_rec,
    }
    results.update(Ps)
    results.update(Rs)
    results.update(Fs)
    print(results)
    return results


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Evaluation script parameters")
    parser.add_argument("--gt_mesh", required=True, type=str)
    parser.add_argument("--recon_mesh", required=True, type=str)
    parser.add_argument("--transform", default="", type=str)
    args = parser.parse_args()
    transform = np.eye(4)
    if args.transform != "":
        transform = np.loadtxt(args.transform)
    eval_pcd(args.recon_mesh, args.gt_mesh, transform)
