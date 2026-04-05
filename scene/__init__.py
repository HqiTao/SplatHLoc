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
import random
import json
import torch
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from torch.utils.data import Dataset, DataLoader
from utils.camera_utils import loadCam


class Scene:
    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], num=-1, images_to_read=None, preload_cameras=True): 
        self.args = args
        self.model_path = args.model_path
        self.source_path = args.source_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.feature_type = args.feature_type
        self.longest_edge = args.longest_edge
        self.resolution_scales = resolution_scales
        self.preload_cameras = preload_cameras
        self.shuffle = shuffle

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.feature_type, args.images, args.eval, images_to_read = images_to_read)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        if preload_cameras:
            for resolution_scale in resolution_scales:
                if num != -1:
                    print("Loading Train Cameras")
                    step_train = len(scene_info.train_cameras) // num
                    self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras[::step_train], resolution_scale, args) if not args.eval else []
                    print("Loading Test Cameras")
                    step_test = len(scene_info.test_cameras) // num
                    self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras[::step_test], resolution_scale, args)
                else:
                    print("Loading Train Cameras")
                    self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args) if not args.eval else []
                    print("Loading Test Cameras")
                    self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                "point_cloud",
                                                "iteration_" + str(self.loaded_iter),
                                                "point_cloud.ply"))
            if hasattr(gaussians, 'feat_encoder'):
                weights_path = os.path.join(self.model_path,f"chkpnt{self.loaded_iter}.pth")
                state_dict = torch.load(weights_path, map_location="cpu")
                gaussians.feat_encoder.load_state_dict(state_dict)
                gaussians.feat_encoder.cuda()
                gaussians.feat_encoder.eval()
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, scene_info.loc_feature_dim, args.speedup) 
        
        self.scene_info = scene_info

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        if self.preload_cameras:
            return self.train_cameras[scale]
        else:
            return DataLoader(SceneDataset(self, split='train'), batch_size=None, shuffle=self.shuffle, pin_memory=True, num_workers=4)

    def getTestCameras(self, scale=1.0):
        if self.preload_cameras:
            return self.test_cameras[scale]
        else:
            return DataLoader(SceneDataset(self, split='test'), batch_size=None, shuffle=self.shuffle, pin_memory=True, num_workers=4)
    

class SceneDataset(Dataset):

    def __init__(self, scene: Scene, split='train'):
        self.scene = scene
        self.split = split

    def __getitem__(self, index):
        if self.split == 'train':
            camera = loadCam(self.scene.args, index, self.scene.scene_info.train_cameras[index], self.scene.resolution_scales[0])
        else:
            camera = loadCam(self.scene.args, index, self.scene.scene_info.test_cameras[index], self.scene.resolution_scales[0])
        return camera

    def __len__(self):
        return len(self.scene.scene_info.train_cameras) if self.split == 'train' else len(self.scene.scene_info.test_cameras)