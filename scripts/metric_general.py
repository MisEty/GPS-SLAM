from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips, init_lpips_module
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    depths = []
    image_names = []
    rgb_dir = os.path.join(gt_dir, "camera")
    depth_dir = os.path.join(gt_dir, "depth")
    for fname in sorted(os.listdir(rgb_dir)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            gt = Image.open(Path(rgb_dir) / fname)
            gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :])
            image_names.append(fname)
    for fname in sorted(os.listdir(renders_dir)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            render = Image.open(renders_dir / fname)
            renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :])
    if len(renders) != len(gts):
        print("[ERROR] renders size != gts size!")
        print(len(gts), len(renders))
        return [], [], []
    for fname in sorted(os.listdir(depth_dir)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            depth = Image.open(Path(depth_dir) / fname)
            depths.append(tf.to_tensor(depth).unsqueeze(0))
    return renders, gts, depths, image_names


def evaluate(renders_dir, gt_dir, depth_mask=True):
    full_dict = {}
    per_view_dict = {}
    if depth_mask:
        print("ignore invalid depth!!!")

    try:
        print("Processing images from:", renders_dir, gt_dir)
        full_dict = {}
        per_view_dict = {}

        renders, gts, depths, image_names = readImages(Path(renders_dir), Path(gt_dir))

        ssims = []
        psnrs = []
        lpipss = []
        lpips_modeule = init_lpips_module(torch.device("cuda:0"), net_type="vgg")
        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            render_img = renders[idx].cuda()
            gt_img = gts[idx].cuda()
            depth_img = depths[idx].cuda()
            if depth_mask:
                render_img = render_img * (depth_img > 0)
                gt_img = gt_img * (depth_img > 0)
            psnr_value = psnr(render_img, gt_img)
            ssim_value = ssim(render_img, gt_img)
            lpips_value = lpips_modeule(render_img, gt_img)
            ssims.append(ssim_value)
            psnrs.append(psnr_value)
            lpipss.append(lpips_value)

        print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
        print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
        print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
        print("")

        full_dict.update(
            {
                "SSIM": torch.tensor(ssims).mean().item(),
                "PSNR": torch.tensor(psnrs).mean().item(),
                "LPIPS": torch.tensor(lpipss).mean().item(),
            }
        )
        per_view_dict.update(
            {
                "SSIM": {name: ssim.item() for ssim, name in zip(ssims, image_names)},
                "PSNR": {name: psnr.item() for psnr, name in zip(psnrs, image_names)},
                "LPIPS": {name: lp.item() for lp, name in zip(lpipss, image_names)},
            }
        )

        with open(os.path.join(renders_dir, "results.json"), "w") as fp:
            json.dump(full_dict, fp, indent=True)
        with open(os.path.join(renders_dir, "per_view.json"), "w") as fp:
            json.dump(per_view_dict, fp, indent=True)
    except Exception as e:
        print(
            f"Unable to compute metrics for render path {renders_dir}. Error: {str(e)}"
        )


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Evaluation script parameters")
    parser.add_argument(
        "--gt_path",
        required=True,
        type=str,
        help="Path to the directory containing gt images",
    )
    parser.add_argument(
        "--render_path",
        required=True,
        type=str,
        help="Path to the directory containing render images",
    )
    parser.add_argument(
        "--depth_mask",
        action="store_true",
        help="Use depth mask if provided",
    )
    args = parser.parse_args()

    # Pass the depth_mask argument to the evaluate function
    evaluate(args.render_path, args.gt_path, depth_mask=args.depth_mask)
