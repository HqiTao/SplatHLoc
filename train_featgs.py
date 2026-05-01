#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint

import torch
import pickle
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams
from gaussian_renderer import render_gsplat
from scene import Scene
from utils.general_utils import safe_state, seed_everything
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import torch.nn.functional as F

from encoders.feature_extractor import FeatureExtractor


def training(
    dataset,
    opt,
    testing_iterations,
    saving_iterations,
    checkpoint,
):
    print(opt)
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    print("Feature type:", dataset.feature_type)
    print("Gaussian type:", dataset.gaussian_type)

    if dataset.gaussian_type == "featgs":
        from scene.gaussian_model import FeatureGaussianModel

        if dataset.feature_type == 'sp':
            feature_dim = 256
        elif dataset.feature_type == 'dinov2':
            feature_dim = 768
        gaussians = FeatureGaussianModel(dataset.sh_degree, feature_dim)
    else:
        raise ValueError("Gaussian type not supported")

    scene = Scene(dataset, gaussians)
    # scene = Scene(dataset, gaussians, num=10)

    # load masks
    masks = None
    if os.path.exists(os.path.join(dataset.source_path, dataset.images, "masks.pkl")):
        print(
            "Loading masks from",
            os.path.join(dataset.source_path, dataset.images, "masks.pkl"),
        )
        masks = pickle.load(
            open(os.path.join(dataset.source_path, dataset.images, "masks.pkl"), "rb")
        )

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

    feature_extractor = FeatureExtractor(dataset.feature_type).cuda().eval()

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Feature Gaussian")
    first_iter += 1

    # Training loop
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        render_pkg = render_gsplat(
            viewpoint_cam,
            gaussians,
            background,
            rgb_only=False,
            norm_feat_bf_render=dataset.norm_before_render,
            longest_edge=dataset.longest_edge,
            rasterize_mode="antialiased",
        )

        feature_map_render, image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["feature_map"],
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Loss
        # make sure ground truth image is the same size as the rendered image
        original_image = viewpoint_cam.original_image.cuda()
        gt_image = F.interpolate(
            original_image.unsqueeze(0),
            size=(image.shape[1], image.shape[2]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        if masks is not None:
            # use mask
            obj_mask = masks[viewpoint_cam.image_name][0].cuda()[None]
            sky_mask = masks[viewpoint_cam.image_name][1].cuda()[None]
            distort_mask = masks[viewpoint_cam.image_name][2].cuda()[None]

            # mask obj and distort border
            mask = obj_mask & distort_mask

            image = image * mask
            gt_image = gt_image * mask
            # mask sky
            gt_image[sky_mask.repeat(3, 1, 1) == False] = 1  # 全白

        Ll1 = l1_loss(image, gt_image)

        if feature_map_render is not None:
            feature_map_high = gaussians.feat_encoder(feature_map_render.unsqueeze(0))
            # print(feature_map_high.shape)
            # resize feature map to gt size
            feature_map = F.interpolate(
                feature_map_high,
                size=(args.feature_resolution[0], args.feature_resolution[1]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            feature_map = F.normalize(feature_map, p=2, dim=0)

            gt_feature_map = feature_extractor(original_image[None])["feature_map"][0]
            # resize feature map to gt size
            gt_feature_map = F.interpolate(
                gt_feature_map.unsqueeze(0),
                size=(args.feature_resolution[0], args.feature_resolution[1]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            gt_feature_map = F.normalize(gt_feature_map, p=2, dim=0)

            # mask feature map
            if masks is not None:
                feature_map_mask = (
                    F.interpolate(
                        mask[None].float(),
                        size=(gt_feature_map.shape[1], gt_feature_map.shape[2]),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                    > 0.5
                )
                feature_map *= feature_map_mask
                gt_feature_map *= feature_map_mask

            Ll1_feature = l1_loss(feature_map, gt_feature_map)
        else:
            Ll1_feature = torch.tensor(0.0, device="cuda")

        # overall loss
        loss = (
            (1.0 - opt.lambda_dssim) * Ll1
            + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            + 1.0 * Ll1_feature
        )

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer,
                iteration,
                Ll1,
                Ll1_feature,
                loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                background,
                dataset,
                feature_extractor=feature_extractor,
                masks=masks,
            )
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                torch.save(gaussians.feat_encoder.state_dict(), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats_gsplat(
                    viewspace_point_tensor,
                    visibility_filter,
                    image.shape[2],
                    image.shape[1],
                )

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    print(args.model_path)

    # Set up output folder
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    Ll1_feature,
    loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    background,
    dataset,
    feature_extractor,
    masks=None,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar(
            "train_loss_patches/l1_loss_feature", Ll1_feature.item(), iteration
        )
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)
        tb_writer.add_scalar(
            "total_points", scene.gaussians.get_xyz.shape[0], iteration
        )

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(5, 30, 5)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                l1_feature_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    render_pkg = render_gsplat(
                        viewpoint,
                        scene.gaussians,
                        background,
                        rgb_only=False,
                        norm_feat_bf_render=dataset.norm_before_render,
                        longest_edge=dataset.longest_edge,
                        rasterize_mode="antialiased",
                    )

                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    feature_map_render = render_pkg["feature_map"]

                    original_image = viewpoint.original_image.cuda()

                    gt_image = F.interpolate(
                        original_image.unsqueeze(0),
                        size=(image.shape[1], image.shape[2]),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                    gt_feature_map = feature_extractor(original_image[None])[
                        "feature_map"
                    ][0]

                    if masks is not None:
                        mask = masks[viewpoint.image_name][0].cuda()[None]
                        sky_mask = masks[viewpoint.image_name][1].cuda()[None]
                        distort_mask = masks[viewpoint.image_name][2].cuda()[None]
                        mask = mask & distort_mask

                    if feature_map_render is not None:
                        feature_map_high = scene.gaussians.feat_encoder(feature_map_render.unsqueeze(0))
                        # resize feature map to gt size
                        feature_map = F.interpolate(
                            feature_map_high,
                            size=(feature_map_render.shape[1], feature_map_render.shape[2]),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)
                        feature_map = F.normalize(feature_map, p=2, dim=0)

                        gt_feature_map = F.interpolate(
                            gt_feature_map.unsqueeze(0),
                            size=(feature_map.shape[1], feature_map.shape[2]),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)
                        gt_feature_map = F.normalize(gt_feature_map, p=2, dim=0)
                        if masks is not None:
                            feature_map_mask_float = F.interpolate(
                                mask[None].float(),
                                size=(gt_feature_map.shape[1], gt_feature_map.shape[2]),
                                mode="bilinear",
                                align_corners=False,
                            ).squeeze(0)
                            feature_map_mask = feature_map_mask_float > 0.5
                            feature_map_loss = feature_map * feature_map_mask
                            gt_feature_map_loss = gt_feature_map * feature_map_mask
                        else:
                            feature_map_loss = feature_map
                            gt_feature_map_loss = gt_feature_map
                        l1_feature_test += (
                            l1_loss(feature_map_loss, gt_feature_map_loss)
                            .mean()
                            .double()
                        )

                    if masks is not None:
                        image_loss = image * mask
                        gt_image_loss = gt_image * mask
                        # mask sky
                        gt_image_loss[sky_mask.repeat(3, 1, 1) == False] = 1  # 全白
                    else:
                        image_loss = image
                        gt_image_loss = gt_image

                    l1_test += l1_loss(image_loss, gt_image_loss).mean().double()
                    psnr_test += psnr(image_loss, gt_image_loss).mean().double()

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )
                            if masks is not None:
                                tb_writer.add_images(
                                    config["name"]
                                    + "_view_{}/mask".format(viewpoint.image_name),
                                    mask[None],
                                    global_step=iteration,
                                )
                                tb_writer.add_images(
                                    config["name"]
                                    + "_view_{}/sky_mask".format(viewpoint.image_name),
                                    sky_mask[None],
                                    global_step=iteration,
                                )

                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                l1_feature_test /= len(config["cameras"])
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {} feature L1 {}".format(
                        iteration, config["name"], l1_test, psnr_test, l1_feature_test
                    )
                )
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss_feature",
                        l1_feature_test,
                        iteration,
                    )

        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram", scene.gaussians.get_opacity, iteration
            )
            tb_writer.add_scalar(
                "total_points", scene.gaussians.get_xyz.shape[0], iteration
            )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    seed_everything(2025)
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)

    parser.add_argument("--detect_anomaly", action="store_true", default=False)

    parser.add_argument("--train_detector", action="store_true", default=False)

    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--feature_resolution", type=int, nargs=2, default=[60, 80])

    args = parser.parse_args(sys.argv[1:])

    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.start_checkpoint,
    )

    # All done
    print("\nTraining complete.")
