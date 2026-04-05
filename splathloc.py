import datetime
import json
import os, time
import faiss
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from encoders.feature_extractor import FeatureExtractor
from gaussian_renderer import render_from_pose_gsplat
from scene import Scene
from scene.gaussian_model import GaussianModel, GaussianModel_2dgs
from utils.general_utils import seed_everything
from utils.graphics_utils import fov2focal
from utils.pose_utils import cal_pose_error, solve_pose
from JamMa.src.config.default import get_cfg_defaults
from JamMa.src.lightning.lightning_jamma import PL_JamMa
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
from vpr_model.vpr import get_trained_mixvpr
from scipy.spatial.transform import Rotation as R

# TODO use interpolate
def lift_2d_to_3d(points2d, intrinsic, Twc, depth_map):
    """
    points2d: tensor [N, 2]
    intrinsic: tensor [3, 3]
    Twc: tensor [4, 4]
    depth_map: tensor [H, W]
    """
    device = points2d.device
    depth_idx = points2d.long()
    points2d = points2d + 0.5
    points2d_homo = torch.cat(
        [points2d, torch.ones((points2d.shape[0], 1), device=device)], dim=1
    )
    points3d_camera = (
        torch.inverse(intrinsic)
        @ points2d_homo.T
        * depth_map[depth_idx[:, 1], depth_idx[:, 0]]
    )  # [3, N]
    points3d_camera_homo = torch.cat(
        [
            points3d_camera,
            torch.ones((1, points3d_camera.shape[-1]), device=device),
        ],
        dim=0,
    )  # [4, N]
    points3d_world = Twc @ points3d_camera_homo  # [4, N]
    points3d = points3d_world.T[:, :3]
    return points3d


def mnn_match(corr_matrix, thr=-1):
    """
    corr_matrix: torch.Tensor, shape (B, N, M)
    """
    mask = corr_matrix > thr
    mask = (
        mask
        * (corr_matrix == corr_matrix.max(dim=-1, keepdim=True)[0])
        * (corr_matrix == corr_matrix.max(dim=-2, keepdim=True)[0])
    )
    b_ids, i_ids, j_ids = torch.where(mask)
    return b_ids.squeeze(), i_ids.squeeze(), j_ids.squeeze()


def dual_softmax(corr_matrix, temp=1):
    corr_matrix = corr_matrix / temp
    corr_matrix = F.softmax(corr_matrix, dim=-2) * F.softmax(corr_matrix, dim=-1)
    return corr_matrix


def get_intrinsic(fovx, fovy, width, height):
    focalX = fov2focal(fovx, width)
    focalY = fov2focal(fovy, height)
    K = np.array(
        [
            [focalX, 0.0, width / 2],
            [0.0, focalY, height / 2],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return K


def add_random_transform(pose_w2c, rot_deg_range=5, trans_range=0.5):
    pose_w2c = pose_w2c.copy()

    c2w = np.linalg.inv(pose_w2c)

    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle_deg = np.random.uniform(-rot_deg_range, rot_deg_range)
    rotation = R.from_rotvec(np.deg2rad(angle_deg) * axis)
    rot_matrix = rotation.as_matrix()

    delta_t = np.random.uniform(-trans_range, trans_range, size=(3,))

    delta_transform = np.eye(4)
    delta_transform[:3, :3] = rot_matrix
    delta_transform[:3, 3] = delta_t

    c2w_new = c2w @ delta_transform
    pose_w2c_new = np.linalg.inv(c2w_new)

    return pose_w2c_new.astype(np.float32)


class SplatHLoc:
    def __init__(self, gaussians, config, match_config, train_cameras):
        self.gaussians = gaussians
        self.config = config

        self.feature_extractor = FeatureExtractor(config["feature_type"]).cuda().eval()
        self.longest_edge = config["longest_edge"]

        self.features_dim = 4096
        self.vpr_model = get_trained_mixvpr().eval().cuda()
        self.train_vpr_feats, self.train_poses = self.get_train_pose_render(train_cameras)

        self.extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
        self.matcher_init = LightGlue(features='superpoint').eval().cuda()

        self.matcher = PL_JamMa(match_config, pretrained_ckpt='official')
        self.matcher.backbone.to("cuda").eval()
        self.matcher.matcher.to("cuda").eval()

        self.inlier_count = 0

    @torch.no_grad()
    def get_train_pose_render(self, train_cameras):
        poses = []
        num = len(train_cameras)
        vpr_features = np.empty((num, self.features_dim), dtype="float32")
        print(f"There are {num} images in the anchor image set.")

        for idx, camera_info in enumerate(tqdm(train_cameras, desc="VPR Database")):
            image = camera_info.original_image
            image = F.interpolate(image.unsqueeze(0), size=(320, 320), mode='bilinear', align_corners=False)

            pose = camera_info.world_view_transform.transpose(0, 1).cpu().numpy()
            poses.append(pose)
            feature = self.vpr_model(image.cuda()).cpu().numpy()
            vpr_features[idx, :] = feature

        return vpr_features, poses

    @torch.no_grad()
    def localize(self, query_image, fovx, fovy):
        """
        image: torch.Tensor, shape (3, H, W)
        """
        t0 = time.time()

        # Get feature
        query_coarse_feature_map = self.get_feature_map(query_image)

        t1 = time.time()
        get_feature_time = t1 - t0

        # Init loc
        t2 = time.time()

        Hc, Wc = query_coarse_feature_map.shape[-2:]
        H = Hc * 8
        W = Wc * 8

        if query_image.shape[-2] != H or query_image.shape[-1] != W:
            query_image = F.interpolate(
                query_image.unsqueeze(0),           
                size=(H, W),             
                mode='bilinear',            
                align_corners=False,         
                antialias=True            
            ).squeeze(0)

        images = [query_image]
        init_result = self.init_loc(query_coarse_feature_map, images, fovx, fovy)
        sparse_pose = init_result["pose_w2c"]
        self.inlier_count = init_result["inliers"]

        t3 = time.time()
        init_loc_time = t3 - t2

        # refine loc
        t4 = time.time()

        pose_w2c = sparse_pose
        dense_results = []
        for iter in range(self.config["dense"]["iters"]):
            dense_result = self.loc_dense(
                query_image, query_coarse_feature_map, pose_w2c, fovx, fovy
            )
            pose_w2c = dense_result["pose_w2c"]
            self.inlier_count = dense_result["inliers"]
            
            dense_results.append(dense_result)
        
        t5 = time.time()
        refine_loc_time = t5 - t4

        total_time = get_feature_time + init_loc_time + refine_loc_time

        return {"sparse": init_result, 
                "dense": dense_results,
                "timing": {
                    "get_feature": get_feature_time,
                    "init_loc": init_loc_time,
                    "refine_loc": refine_loc_time,
                    "total": total_time
                }}

    @torch.no_grad()
    def init_loc(self, query_coarse_feature_map, images, fovx, fovy):
        query = images[0].unsqueeze(0)
        Hf, Wf = query.shape[-2:]
        K = get_intrinsic(fovx, fovy, Wf, Hf)

        query = F.interpolate(query, size=(320, 320), mode='bilinear', align_corners=False)
        query_feature = self.vpr_model(query).cpu().numpy()
        database_features = self.train_vpr_feats

        faiss_index = faiss.IndexFlatL2(self.features_dim)
        faiss_index.add(database_features)
        del database_features

        distances, predictions = faiss_index.search(query_feature, config["init"]["topk"])
        predictions = predictions[0]

        idx_img = -1
        num_inliers = 0
        best_img_id = 0
        
        data = {
            "pose_w2c": [],
            "render": [],
            "depth": []
        }

        for i in range(0, len(predictions), config["init"]["interval"]):
            prediction = predictions[i]
            pose_w2c = self.train_poses[prediction]
            idx_img += 1

            render_pkg = render_from_pose_gsplat(
                self.gaussians,
                torch.tensor(pose_w2c, device="cuda"),
                fovx,
                fovy,
                Wf,
                Hf,
                rgb_only=True,
                render_mode="RGB+ED",
                norm_feat_bf_render=self.config["dense"]["norm_before_render"],
                rasterize_mode="antialiased",
            )
            images.append(render_pkg["render"])
            depth = render_pkg["depth"].squeeze()

            render = render_pkg["render"]

            data["pose_w2c"].append(pose_w2c)
            data["render"].append(render)
            data["depth"].append(depth) 

            # match the features
            feats0 = self.extractor.extract(images[0])
            feats1 = self.extractor.extract(render)

            matches01 = self.matcher_init({'image0': feats0, 'image1': feats1})
            feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
            matches = matches01['matches']  # indices with shape (K,2)
            points0 = feats0['keypoints'][matches[..., 0]].cpu().numpy()
            points1 = feats1['keypoints'][matches[..., 1]].cpu().numpy()

            query_p2d  = torch.from_numpy(points0).float().to("cuda")
            rendered_p2d = torch.from_numpy(points1).float().to("cuda")

            pose_c2w = np.linalg.inv(pose_w2c)

            p3d = lift_2d_to_3d(
                rendered_p2d,
                torch.tensor(K, device="cuda"),
                torch.tensor(pose_c2w, device="cuda"),
                depth
            )

            # Solve pose
            query_p2d = query_p2d.cpu().numpy()
            p3d = p3d.cpu().numpy()

            _, inliers = solve_pose(
                query_p2d + 0.5,
                p3d,
                K,
                self.config["dense"]["solver"],
                self.config["dense"]["reprojection_error"],
                self.config["dense"]["confidence"],
                self.config["dense"]["max_iterations"],
                self.config["dense"]["min_iterations"],
            )

            if inliers.shape[0] > num_inliers:
                num_inliers = inliers.shape[0]
                best_img_id = idx_img

            if num_inliers > config["init"]["iniler_threshold_1"]:
                break

        pose_init =data["pose_w2c"][best_img_id]

        if num_inliers < config["init"]["iniler_threshold_2"]:
            num_perturbed_pose = config["init"]["num_perturbed"]
            base_pose = pose_init.copy()
            perturbed_poses = []
            vpr_features = np.empty((num_perturbed_pose, self.features_dim), dtype="float32")
            for i in range(num_perturbed_pose):
                pose_virtual = add_random_transform(base_pose, rot_deg_range=config["init"]["rot_deg"], \
                                                    trans_range=config["init"]["trans"])
                render_pkg = render_from_pose_gsplat(
                    self.gaussians,
                    torch.tensor(pose_virtual, dtype=torch.float32, device="cuda"),
                    fovx,
                    fovy,
                    320,
                    320,
                    rgb_only=True,
                    render_mode="RGB",
                    norm_feat_bf_render=self.config["dense"]["norm_before_render"],
                    rasterize_mode="antialiased",
                )
                render = render_pkg["render"].unsqueeze(0)
                feature = self.vpr_model(render).cpu().numpy()
                vpr_features[i, :] = feature
                perturbed_poses.append(pose_virtual)

            new_faiss_index = faiss.IndexFlatL2(self.features_dim)
            new_faiss_index.add(vpr_features)
            del vpr_features
            distances, predictions = new_faiss_index.search(query_feature, 5)
            predictions = predictions[0]

            for j in range(0, len(predictions)):
                prediction = predictions[j]
                pose_w2c = perturbed_poses[prediction]

                render_pkg = render_from_pose_gsplat(
                    self.gaussians,
                    torch.tensor(pose_w2c, device="cuda"),
                    fovx,
                    fovy,
                    Wf,
                    Hf,
                    rgb_only=True,
                    render_mode="RGB+ED",
                    norm_feat_bf_render=self.config["dense"]["norm_before_render"],
                    rasterize_mode="antialiased",
                )

                depth = render_pkg["depth"].squeeze()
                render = render_pkg["render"]

                # match the features
                feats0 = self.extractor.extract(images[0])
                feats1 = self.extractor.extract(render)

                matches01 = self.matcher_init({'image0': feats0, 'image1': feats1})
                feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
                matches = matches01['matches']  # indices with shape (K,2)
                points0 = feats0['keypoints'][matches[..., 0]].cpu().numpy()
                points1 = feats1['keypoints'][matches[..., 1]].cpu().numpy()

                query_p2d  = torch.from_numpy(points0).float().to("cuda")
                rendered_p2d = torch.from_numpy(points1).float().to("cuda")

                pose_c2w = np.linalg.inv(pose_w2c)

                p3d = lift_2d_to_3d(
                    rendered_p2d,
                    torch.tensor(K, device="cuda"),
                    torch.tensor(pose_c2w, device="cuda"),
                    depth
                )

                # Solve pose
                query_p2d = query_p2d.cpu().numpy()
                p3d = p3d.cpu().numpy()

                _, inliers = solve_pose(
                    query_p2d + 0.5,
                    p3d,
                    K,
                    self.config["dense"]["solver"],
                    self.config["dense"]["reprojection_error"],
                    self.config["dense"]["confidence"],
                    self.config["dense"]["max_iterations"],
                    self.config["dense"]["min_iterations"],
                )

                if inliers.shape[0] > num_inliers:
                    pose_init = pose_w2c
                    num_inliers = inliers.shape[0]
                    print("successful fine retrieval!")
                    

        Hc, Wc = query_coarse_feature_map.shape[-2:]

        render_pkg = render_from_pose_gsplat(
            self.gaussians,
            torch.tensor(pose_init, device="cuda"),
            fovx,
            fovy,
            Wf,
            Hf,
            render_mode="RGB+ED",
            norm_feat_bf_render=self.config["dense"]["norm_before_render"],
            rasterize_mode="antialiased",
        )
        
        fine_rendered_feature_map = render_pkg["feature_map"]
        depth = render_pkg["depth"].squeeze()
        if hasattr(gaussians, 'feat_encoder'):
            fine_rendered_feature_map = gaussians.feat_encoder(fine_rendered_feature_map.unsqueeze(0)).squeeze(0)
        coarse_rendered_feature_map = F.interpolate(
            fine_rendered_feature_map[None],
            size=(Hc, Wc),
            mode="bilinear",
            align_corners=False,
        )[0]
        coarse_rendered_feature_map = F.normalize(coarse_rendered_feature_map, dim=0)

        # coarse match
        C = self.feature_extractor.feature_dim
        coarse_corr_matrix = torch.matmul(
            query_coarse_feature_map.permute(1, 2, 0).reshape(1, -1, C),
            coarse_rendered_feature_map.reshape(1, C, -1),
        )  # 1, N, M

        coarse_corr_matrix = dual_softmax(
            coarse_corr_matrix, temp=self.config["dense"]["coarse_dual_softmax_temp"]
        )

        c_b_ids, c_i_ids, c_j_ids = mnn_match(
            coarse_corr_matrix, thr=self.config["dense"]["coarse_threshold"]
        )

        if c_i_ids.dim() == 0:
            print("[skip] Failed in coarse match")
            return {"pose_w2c": pose_w2c, "inliers": 0}
        elif c_i_ids.shape[0] < 3:
            print("[skip] Failed in coarse match")
            return {"pose_w2c": pose_w2c, "inliers": 0}
        
        # fine match
        K_tensor = torch.tensor(K, device="cuda")
        B = 1
        h_8 = Hf / 8
        w_8 = Wf / 8
        data = {
            'bs': B,
            'c': C,
            'hw_8': h_8 * w_8,
            'imagec_0': images[0].unsqueeze(0),
            'imagec_1': render_pkg["render"].unsqueeze(0),
            'K0': K_tensor.unsqueeze(0),  # (3, 3)
            'K1': K_tensor.unsqueeze(0),
        }

        with torch.autocast(enabled=False, device_type='cuda'):
            self.matcher.backbone(data)
            data.update({
                'hw0_i': data['imagec_0'].shape[2:],
                'hw1_i': data['imagec_1'].shape[2:],
                'hw0_c': [data['h_8'], data['w_8']],
                'hw1_c': [data['h_8'], data['w_8']],
            })
            self.matcher.matcher.coarse_match(data)
 
            mkpts0_c = torch.stack(
                [c_i_ids % data['hw0_c'][1], torch.div(c_i_ids, data['hw0_c'][1], rounding_mode='trunc')],
                dim=1) * 8
            mkpts1_c = torch.stack(
                [c_j_ids % data['hw1_c'][1], torch.div(c_j_ids, data['hw1_c'][1], rounding_mode='trunc')],
                dim=1) * 8

            data.update({
                'b_ids': c_b_ids, 
                'i_ids': c_i_ids, 
                'j_ids': c_j_ids,
                'mkpts0_c': mkpts0_c,
                'mkpts1_c': mkpts1_c
            })
            feat_f0_unfold, feat_f1_unfold = self.matcher.matcher.fine_preprocess(data, None)

            self.matcher.matcher.fine_matching(feat_f0_unfold.transpose(1, 2), feat_f1_unfold.transpose(1, 2), data)

        pose_c2w = np.linalg.inv(pose_init)

        rendered_p2d = data['mkpts1_f']
        p3d = lift_2d_to_3d(rendered_p2d, K_tensor, torch.tensor(pose_c2w, device="cuda"), depth)

        # Solve pose
        query_p2d = data['mkpts0_f'].cpu().numpy()
        p3d = p3d.cpu().numpy()

        pose_w2c, inliers = solve_pose(
            query_p2d + 0.5,
            p3d,
            K,
            self.config["dense"]["solver"],
            self.config["dense"]["reprojection_error"],
            self.config["dense"]["confidence"],
            self.config["dense"]["max_iterations"],
            self.config["dense"]["min_iterations"],
        )

        return {
            "pose_w2c": pose_w2c,
            "inliers": inliers.shape[0],
        }

    @torch.no_grad()
    def loc_dense(self, query, coarse_query_feature_map, pose_w2c, fovx, fovy):
        """
        coarse_feature_map: torch.Tensor, shape (C, H, W)
        fine_feature_map: torch.Tensor, shape (C, H, W)
        """
        Hc, Wc = coarse_query_feature_map.shape[-2:]
        C = self.feature_extractor.feature_dim
        K = get_intrinsic(fovx, fovy, Wc * 8, Hc * 8)

        render_pkg = render_from_pose_gsplat(
            self.gaussians,
            torch.tensor(pose_w2c, device="cuda"),
            fovx,
            fovy,
            Wc * 8,
            Hc * 8,
            render_mode="RGB+ED",
            norm_feat_bf_render=self.config["dense"]["norm_before_render"],
            rasterize_mode="antialiased",
        )

        depth = render_pkg["depth"].squeeze()

        fine_rendered_feature_map = render_pkg["feature_map"]
        if (fine_rendered_feature_map == 0).all():
            print("[skip] Rendered feature map is all zero")
            return {"pose_w2c": pose_w2c, "inliers": self.inlier_count}

        if hasattr(gaussians, 'feat_encoder'):
            fine_rendered_feature_map = gaussians.feat_encoder(fine_rendered_feature_map.unsqueeze(0)).squeeze(0)
        
        coarse_rendered_feature_map = F.interpolate(
            fine_rendered_feature_map[None],
            size=(Hc, Wc),
            mode="bilinear",
            align_corners=False,
        )[0]
        coarse_rendered_feature_map = F.normalize(coarse_rendered_feature_map, dim=0)

        # coarse match
        coarse_corr_matrix = torch.matmul(
            coarse_query_feature_map.permute(1, 2, 0).reshape(1, -1, C),
            coarse_rendered_feature_map.reshape(1, C, -1),
        )  # 1, N, M

        coarse_corr_matrix = dual_softmax(
            coarse_corr_matrix, temp=self.config["dense"]["coarse_dual_softmax_temp"]
        )

        c_b_ids, c_i_ids, c_j_ids = mnn_match(
            coarse_corr_matrix, thr=self.config["dense"]["coarse_threshold"]
        )

        if c_i_ids.dim() == 0:
            print("[skip] Failed in coarse match")
            return {"pose_w2c": pose_w2c, "inliers": self.inlier_count}
        elif c_i_ids.shape[0] < 10:
            print("[skip] Failed in coarse match")
            return {"pose_w2c": pose_w2c, "inliers": self.inlier_count}
        
        # fine match
        K_tensor = torch.tensor(K, device="cuda")
        B = 1
        h_8 = Hc
        w_8 = Wc
        data = {
            'bs': B,
            'c': C,
            'hw_8': h_8 * w_8,
            'imagec_0': query.unsqueeze(0),
            'imagec_1': render_pkg["render"].unsqueeze(0),
            'K0': K_tensor.unsqueeze(0),  # (3, 3)
            'K1': K_tensor.unsqueeze(0),
        }

        with torch.autocast(enabled=False, device_type='cuda'):
            self.matcher.backbone(data)
            data.update({
                'hw0_i': data['imagec_0'].shape[2:],
                'hw1_i': data['imagec_1'].shape[2:],
                'hw0_c': [data['h_8'], data['w_8']],
                'hw1_c': [data['h_8'], data['w_8']],
            })
            self.matcher.matcher.coarse_match(data)
 
            mkpts0_c = torch.stack(
                [c_i_ids % data['hw0_c'][1], torch.div(c_i_ids, data['hw0_c'][1], rounding_mode='trunc')],
                dim=1) * 8
            mkpts1_c = torch.stack(
                [c_j_ids % data['hw1_c'][1], torch.div(c_j_ids, data['hw1_c'][1], rounding_mode='trunc')],
                dim=1) * 8

            data.update({
                'b_ids': c_b_ids, 
                'i_ids': c_i_ids, 
                'j_ids': c_j_ids,
                'mkpts0_c': mkpts0_c,
                'mkpts1_c': mkpts1_c
            })
            feat_f0_unfold, feat_f1_unfold = self.matcher.matcher.fine_preprocess(data, None)

            self.matcher.matcher.fine_matching(feat_f0_unfold.transpose(1, 2), feat_f1_unfold.transpose(1, 2), data)

        pose_c2w = np.linalg.inv(pose_w2c)
        rendered_p2d = data['mkpts1_f']
        p3d = lift_2d_to_3d(rendered_p2d, K_tensor, torch.tensor(pose_c2w, device="cuda"), depth)

        # Solve pose
        query_p2d = data['mkpts0_f'].cpu().numpy()
        p3d = p3d.cpu().numpy()

        pose_w2c_new, inliers = solve_pose(
            query_p2d + 0.5,
            p3d,
            K,
            self.config["dense"]["solver"],
            self.config["dense"]["reprojection_error"],
            self.config["dense"]["confidence"],
            self.config["dense"]["max_iterations"],
            self.config["dense"]["min_iterations"],
        )

        if inliers.shape[0] < 10:
            print("[skip] Failed in fine match")
            return {"pose_w2c": pose_w2c, "inliers": self.inlier_count}

        return {
            "pose_w2c": pose_w2c_new,
            "inliers": inliers.shape[0],
        }

    def get_feature_map(self, image):
        """
        image: torch.Tensor, shape (3, H, W)
        """

        # Get feature
        feature_map = self.feature_extractor(image[None])["feature_map"]  # 1, C, H, W

        coarse_feature_map = F.interpolate(
            feature_map, size=self.config["coarse_resolution"], mode="bilinear", align_corners=False
        )[0]
        coarse_feature_map = F.normalize(coarse_feature_map, p=2, dim=0)

        return coarse_feature_map


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--cfg", default=None, type=str)
    parser.add_argument("--prefix", default=None, type=str)
    parser.add_argument(
    '--main_cfg_path', type=str, help='main config path')
    args = get_combined_args(parser)
    # args.eval = True

    if hasattr(args, "prefix"):
        output_path = f"results/{args.prefix}-{args.model_path.replace('/', '_')}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        output_path = f"results/{args.model_path.replace('/', '_')}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print("Output path:", output_path)
    os.makedirs(output_path, exist_ok=True)

    seed_everything(2025)

    # Load feature gaussian scene
    dataset = model.extract(args)
    if dataset.gaussian_type == "3dgs":
        gaussians = GaussianModel(dataset.sh_degree)
    elif dataset.gaussian_type == "2dgs":
        gaussians = GaussianModel_2dgs(dataset.sh_degree)
    elif dataset.gaussian_type == "featgs":
        from scene.gaussian_model import FeatureGaussianModel

        gaussians = FeatureGaussianModel(dataset.sh_degree, high_dim=256)
    else:
        raise ValueError("Gaussian type not supported")

    scene = Scene(
        dataset,
        gaussians,
        load_iteration=args.iteration,
        shuffle=False,
        preload_cameras=False,
    )

    # Set up config
    config = yaml.load(open(args.cfg), Loader=yaml.FullLoader)
        
    config["dense"]["norm_before_render"] = dataset.norm_before_render
    config["feature_type"] = dataset.feature_type
    config["longest_edge"] = dataset.longest_edge
    config["model_path"] = dataset.model_path

    match_config = get_cfg_defaults()
    match_config.merge_from_file(args.main_cfg_path)

    yaml.dump(config, open(os.path.join(output_path, os.path.basename(args.cfg)), "w"))

    test_cameras = scene.getTestCameras()
    train_cameras = scene.getTrainCameras()

    # loc main
    splathloc = SplatHLoc(gaussians, config, match_config, train_cameras)

    results = []
    sparse_aes = []
    sparse_tes = []
    sparse_inliers = []
    dense_aes = []
    dense_tes = []
    dense_inliers = []
    get_feature_times = []
    init_loc_times = []
    refine_loc_times = []
    total_times = []

    for idx, camera_info in enumerate(tqdm(test_cameras, desc="SplatHLoc")):
        print("\nLocalize image:", camera_info.image_name)
        gt_w2c = camera_info.world_view_transform.transpose(0, 1).cpu().numpy()
        query_image = camera_info.original_image.to("cuda")
        fovx = camera_info.FoVx
        fovy = camera_info.FoVy

        # localization
        loc_res = splathloc.localize(query_image, fovx, fovy)

        # evaluation
        sparse_ae, sparse_te = cal_pose_error(loc_res["sparse"]["pose_w2c"], gt_w2c)
        sparse_aes.append(sparse_ae)
        sparse_tes.append(sparse_te)
        sparse_inliers.append(loc_res["sparse"]["inliers"])
        loc_res["image_name"] = camera_info.image_name
        loc_res["sparse_AE"] = sparse_ae
        loc_res["sparse_TE"] = sparse_te

        dense_ae, dense_te = cal_pose_error(loc_res["dense"][-1]["pose_w2c"], gt_w2c) # degree, cm
        dense_aes.append(dense_ae)
        dense_tes.append(dense_te)
        dense_inliers.append(loc_res["dense"][-1]["inliers"])
        print(f"AE: {dense_ae:.3f}deg, TE: {dense_te:.3f}cm, inliers: {loc_res['dense'][-1]['inliers']}")

        loc_res["gt_pose_w2c"] = gt_w2c.tolist()
        loc_res["dense_AE"] = dense_ae
        loc_res["dense_TE"] = dense_te

        get_feature_times.append(loc_res["timing"]["get_feature"])
        init_loc_times.append(loc_res["timing"]["init_loc"])
        refine_loc_times.append(loc_res["timing"]["refine_loc"])
        total_times.append(loc_res["timing"]["total"])

        results.append(loc_res)

    # get summary
    sparse_aes = np.array(sparse_aes)
    sparse_tes = np.array(sparse_tes)
    dense_aes = np.array(dense_aes)
    dense_tes = np.array(dense_tes)

    results_summary = {
        "model_path": dataset.model_path,
        "sparse": {
            "median_ae": np.median(sparse_aes),
            "median_te": np.median(sparse_tes),
            "recall_5m_10d": ((sparse_aes <= 10) & (sparse_tes <= 500)).sum()
            / len(sparse_aes),
            "recall_2m_5d": ((sparse_aes <= 5) & (sparse_tes <= 200)).sum()
            / len(sparse_aes),
            "recall_5cm_5d": ((sparse_aes <= 5) & (sparse_tes <= 5)).sum()
            / len(sparse_aes),
            "recall_2cm_2d": ((sparse_aes <= 2) & (sparse_tes <= 2)).sum()
            / len(sparse_aes),
            "avg_inliers": np.array(sparse_inliers).mean(),
        },
        "dense": {
            "median_ae": np.median(dense_aes),
            "median_te": np.median(dense_tes),
            "recall_5m_10d": ((dense_aes <= 10) & (dense_tes <= 500)).sum()
            / len(dense_aes),
            "recall_2m_5d": ((dense_aes <= 5) & (dense_tes <= 200)).sum()
            / len(dense_aes),
            "recall_5cm_5d": ((dense_aes <= 5) & (dense_tes <= 5)).sum()
            / len(dense_aes),
            "recall_2cm_2d": ((dense_aes <= 2) & (dense_tes <= 2)).sum()
            / len(dense_aes),
            "avg_inliers": np.array(dense_inliers).mean(),
        },
        "timing": {
            "mean_get_feature": np.mean(get_feature_times),
            "mean_init_loc": np.mean(init_loc_times),
            "mean_refine_loc": np.mean(refine_loc_times),
            "mean_total": np.mean(total_times),
        }
    }

    print("Result Summary:")
    print(json.dumps(results_summary, indent=4))

    json.dump(
        results_summary, open(os.path.join(output_path, "summary.json"), "w"), indent=4
    )

    for item in results:
        item["sparse"]["pose_w2c"] = item["sparse"]["pose_w2c"].tolist()
        for dense_item in item["dense"]:
            dense_item["pose_w2c"] = dense_item["pose_w2c"].tolist()
    json.dump(results, open(os.path.join(output_path, "results.json"), "w"), indent=4)

    print("Result are saved in", output_path)
