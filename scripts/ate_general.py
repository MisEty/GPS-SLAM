import numpy as np
import os
from pathlib import Path
from argparse import ArgumentParser


def readPoses(est_dir, gt_dir):
    ests = []
    gts = []
    pose_names = []
    for fname in sorted(os.listdir(gt_dir)):
        if fname.lower().endswith((".txt")):
            gt = np.loadtxt(gt_dir / fname)
            if gt.shape == (4, 4):
                gts.append(gt)
                pose_names.append(fname)
    for fname in sorted(os.listdir(est_dir)):
        if fname.lower().endswith((".txt")):
            est = np.loadtxt(est_dir / fname)
            if est.shape == (4, 4):
                ests.append(est)
    print(len(ests), len(gts))
    if len(ests) != len(gts):
        print("[ERROR] ests size != gts size!")
        return [], [], []
    return ests, gts, pose_names


def align(model, data):
    np.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1).reshape((3, -1))
    data_zerocentered = data - data.mean(1).reshape((3, -1))

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if np.linalg.det(U) * np.linalg.det(Vh) < 0:
        S[2, 2] = -1
    rot = U * S * Vh
    trans = data.mean(1).reshape((3, -1)) - rot * model.mean(1).reshape((3, -1))

    model_aligned = rot * model + trans
    alignment_error = model_aligned - data

    trans_error = np.sqrt(np.sum(np.multiply(alignment_error, alignment_error), 0)).A[0]
    return rot, trans, trans_error


def evaluate(est_dir, gt_dir):
    print("Processing poses from:", est_dir, gt_dir)
    ests, gts, pose_names = readPoses(Path(est_dir), Path(gt_dir))

    est_traj_points = [ests[idx][:3, 3] for idx in range(len(ests))]
    est_traj = np.stack(est_traj_points).T
    gt_traj_points = [gts[idx][:3, 3] for idx in range(len(gts))]
    gt_graj = np.stack(gt_traj_points).T

    _, _, trans_error = align(gt_graj, est_traj)
    avg_trans_error = trans_error.mean()
    print(f"ATE RMSE: {avg_trans_error*100.:.2f}")
    with open(os.path.join(est_dir, "../pose_eval.txt"), "w") as f:
        print(f"ATE RMSE: {avg_trans_error*100.:.2f}", file=f)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Evaluation script parameters")
    parser.add_argument(
        "--gt_path",
        required=True,
        type=str,
        help="Path to the directory containing gt poses",
    )
    parser.add_argument(
        "--est_path",
        required=True,
        type=str,
        help="Path to the directory containing estimated poses",
    )
    args = parser.parse_args()
    evaluate(args.est_path, args.gt_path)
