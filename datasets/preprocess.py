# import some common libraries
import sys

sys.path.append(".")
sys.path.append("submodules/Mask2Former")

import argparse
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from tqdm import tqdm

setup_logger()
setup_logger(name="mask2former")
coco_metadata = MetadataCatalog.get("coco_2017_val_panoptic")

# import project
from mask2former import add_maskformer2_config
from scene.colmap_loader import read_extrinsics_binary, read_extrinsics_text, read_intrinsics_binary, read_intrinsics_text
from utils.image_utils import get_resolution_from_longest_edge


def hist_equalize(image):
    r, g, b = cv2.split(image)
    #
    clahe_b = hist_equalizer.apply(b)
    clahe_g = hist_equalizer.apply(g)
    clahe_r = hist_equalizer.apply(r)

    # merge
    clahe_image_rgb = cv2.merge((clahe_r, clahe_g, clahe_b))
    return clahe_image_rgb


class stuff_masker(torch.nn.Module):
    def __init__(self):
        super(stuff_masker, self).__init__()
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(
            "submodules/Mask2Former/configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml"
        )
        cfg.MODEL.WEIGHTS = "submodules/Mask2Former/model_final_f07440.pkl"
        cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
        cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
        cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
        cfg.freeze()

        predictor = DefaultPredictor(cfg)
        self.predictor = predictor

    def get_stuff_mask(self, image):
        # BGR
        outputs = self.predictor(image)
        stuff_mask = torch.zeros_like(outputs["panoptic_seg"][0])
        for info in outputs["panoptic_seg"][1]:
            if info["isthing"] is False:
                stuff_mask[outputs["panoptic_seg"][0] == info["id"]] = 1
        return stuff_mask

    def get_stuff_and_sky_mask(self, image):
        # BGR
        outputs = self.predictor(image)
        stuff_mask = torch.ones_like(outputs["panoptic_seg"][0], dtype=torch.bool)
        sky_mask = torch.ones_like(outputs["panoptic_seg"][0], dtype=torch.bool)
        for info in outputs["panoptic_seg"][1]:
            if info["isthing"]:
                stuff_mask[outputs["panoptic_seg"][0] == info["id"]] = False
            # mask sky
            if info["category_id"] == 119:
                sky_mask[outputs["panoptic_seg"][0] == info["id"]] = False
        return stuff_mask, sky_mask

    def forward(self, image):
        return self.get_stuff_and_sky_mask(image)


def undistort(distorted_image, camera_matrix, distortion_coeffs):
    # read
    if distorted_image is None:
        raise ValueError("distorted_image is None")
    # undistort
    undistorted_image = cv2.undistort(distorted_image, camera_matrix, distortion_coeffs)
    return undistorted_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, default="")
    parser.add_argument("--images", type=str, default="")
    parser.add_argument("--longest_edge", type=int, default=640)
    parser.add_argument("--output_folder", type=str, default="processed")

    args = parser.parse_args()

    colmap_path = os.path.join(args.source_path, "sparse", "0")
    extrinsics = read_extrinsics_binary(os.path.join(colmap_path, "images.bin"))

    output_path = os.path.join(args.source_path, args.output_folder)
    os.makedirs(output_path, exist_ok=True)

    try:
        cameras_extrinsic_file = os.path.join(colmap_path, "images.bin")
        cameras_intrinsic_file = os.path.join(colmap_path, "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(colmap_path, "images.txt")
        cameras_intrinsic_file = os.path.join(colmap_path, "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    masker = stuff_masker()
    hist_equalizer = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    masks = {}

    for key in tqdm(cam_extrinsics, desc="Prepocessing"):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        image_name = extr.name
        image_output_path = os.path.join(
            args.source_path, args.output_folder, image_name
        )
        os.makedirs(os.path.dirname(image_output_path), exist_ok=True)

        if intr.model == "SIMPLE_RADIAL":
            camera_matrix = np.array(
                [
                    [intr.params[0], 0, intr.params[1]],
                    [0, intr.params[0], intr.params[2]],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )  # focal length and principal point
            distortion_coeffs = np.array([intr.params[3], 0, 0, 0], dtype=np.float32)

        # read image
        image_path = os.path.join(args.source_path, args.images, image_name)
        distorted_image = cv2.imread(image_path)

        # CLAHE hist equalization
        hist_equalized_image = hist_equalize(distorted_image)

        # undistort
        undistorted_image = undistort(
            hist_equalized_image, camera_matrix, distortion_coeffs
        )

        # save processed image
        cv2.imwrite(image_output_path, undistorted_image)

        # generate masks
        image = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1)

        undistort_mask = torch.max(image, dim=0, keepdim=True)[0] > 0

        mask_resolution = get_resolution_from_longest_edge(
            image.shape[1], image.shape[2], args.longest_edge
        )

        image = F.interpolate(
            image.unsqueeze(dim=0).float(),
            size=mask_resolution,
            mode="bilinear",
            align_corners=False,
        ).squeeze(dim=0)
        undistort_mask = (
            F.interpolate(
                undistort_mask.unsqueeze(dim=0).float(),
                size=mask_resolution,
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            > 0.5
        )
        stuff_mask, sky_mask = masker(image.permute(1, 2, 0).numpy()[:, :, ::-1])
        mask = (stuff_mask, sky_mask, undistort_mask)
        masks[image_name] = mask

    pickle.dump(masks, open(os.path.join(output_path, "masks.pkl"), "wb"))
    print("Masks saved to", os.path.join(output_path, "masks.pkl"))
