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


def readImages(input_path):
    # renders_dir = input_path / "raycast_color"
    renders_dir = input_path / "render"
    gt_dir = input_path / "gt"
    renders = []
    gts = []
    image_names = []
    for fname in sorted(os.listdir(gt_dir)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")) and "color" in fname:
            # print(fname[:-19] + ".color.jpg")
            # render = Image.open(renders_dir / (fname[:-20] + ".color.jpg"))
            render = Image.open(renders_dir / fname)
            gt = Image.open(gt_dir / fname)
            renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :])
            gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :])
            image_names.append(fname)
    # print(gts[0].mean(), renders[0].mean())
    return renders, gts, image_names


def evaluate(input_path):
    full_dict = {}
    per_view_dict = {}

    try:
        print("Processing images from:", input_path)
        full_dict[input_path] = {}
        per_view_dict[input_path] = {}

        renders, gts, image_names = readImages(Path(input_path))

        ssims = []
        psnrs = []
        lpipss = []
        lpips_modeule = init_lpips_module(torch.device("cuda:0"), net_type="vgg")
        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            render_img = renders[idx].cuda()
            gt_img = gts[idx].cuda()
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

        full_dict[input_path].update(
            {
                "SSIM": torch.tensor(ssims).mean().item(),
                "PSNR": torch.tensor(psnrs).mean().item(),
                "LPIPS": torch.tensor(lpipss).mean().item(),
            }
        )
        per_view_dict[input_path].update(
            {
                "SSIM": {name: ssim.item() for ssim, name in zip(ssims, image_names)},
                "PSNR": {name: psnr.item() for psnr, name in zip(psnrs, image_names)},
                "LPIPS": {name: lp.item() for lp, name in zip(lpipss, image_names)},
            }
        )

        with open(os.path.join(input_path, "results.json"), "w") as fp:
            json.dump(full_dict[input_path], fp, indent=True)
        with open(os.path.join(input_path, "per_view.json"), "w") as fp:
            json.dump(per_view_dict[input_path], fp, indent=True)
    except Exception as e:
        print(f"Unable to compute metrics for input path {input_path}. Error: {str(e)}")


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Evaluation script parameters")
    parser.add_argument(
        "--input_path",
        "-i",
        required=True,
        type=str,
        help="Path to the directory containing 'gt' and 'renders' subdirectories",
    )
    args = parser.parse_args()
    evaluate(args.input_path)
