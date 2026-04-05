import os
import sys
import collections
import numpy as np
import struct
import argparse
import logging
from tqdm import tqdm
from collections import defaultdict

logging.basicConfig(stream=sys.stdout,
                    format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
Image = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])



def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    """pack and write to a binary file.
    :param fid:
    :param data: data to send, if multiple elements are sent at the same time,
    they should be encapsuled either in a list or a tuple
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    should be the same length as the data list or tuple
    :param endian_character: Any of {@, =, <, >, !}
    """
    if isinstance(data, (list, tuple)):
        bytes = struct.pack(endian_character + format_char_sequence, *data)
    else:
        bytes = struct.pack(endian_character + format_char_sequence, data)
    fid.write(bytes)
    

def write_cameras_text(cameras, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    HEADER = "# Camera list with one line of data per camera:\n" + \
             "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n" + \
             "# Number of cameras: {}\n".format(len(cameras))
    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, cam in cameras.items():
            to_write = [cam.id, cam.model, cam.width, cam.height, *cam.params]
            line = " ".join([str(elem) for elem in to_write])
            fid.write(line + "\n")


def write_cameras_binary(cameras, path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(cameras), "Q")
        for _, cam in cameras.items():
            model_id = CAMERA_MODEL_NAMES[cam.model].model_id
            camera_properties = [cam.id,
                                 model_id,
                                 cam.width,
                                 cam.height]
            write_next_bytes(fid, camera_properties, "iiQQ")
            for p in cam.params:
                write_next_bytes(fid, float(p), "d")
    return cameras


def write_images_text(images, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    if len(images) == 0:
        mean_observations = 0
    else:
        mean_observations = sum((len(img.point3D_ids) for _, img in images.items()))/len(images)
    HEADER = "# Image list with two lines of data per image:\n" + \
             "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n" + \
             "#   POINTS2D[] as (X, Y, POINT3D_ID)\n" + \
             "# Number of images: {}, mean observations per image: {}\n".format(len(images), mean_observations)

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, img in images.items():
            image_header = [img.id, *img.qvec, *img.tvec, img.camera_id, img.name]
            first_line = " ".join(map(str, image_header))
            fid.write(first_line + "\n")

            points_strings = []
            for xy, point3D_id in zip(img.xys, img.point3D_ids):
                points_strings.append(" ".join(map(str, [*xy, point3D_id])))
            fid.write(" ".join(points_strings) + "\n")


def write_images_binary(images, path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(images), "Q")
        for _, img in images.items():
            write_next_bytes(fid, img.id, "i")
            write_next_bytes(fid, img.qvec.tolist(), "dddd")
            write_next_bytes(fid, img.tvec.tolist(), "ddd")
            write_next_bytes(fid, img.camera_id, "i")
            for char in img.name:
                write_next_bytes(fid, char.encode("utf-8"), "c")
            write_next_bytes(fid, b"\x00", "c")
            write_next_bytes(fid, len(img.point3D_ids), "Q")
            for xy, p3d_id in zip(img.xys, img.point3D_ids):
                write_next_bytes(fid, [*xy, p3d_id], "ddq")


def write_points3D_text(points3D, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    if len(points3D) == 0:
        mean_track_length = 0
    else:
        mean_track_length = sum((len(pt.image_ids) for _, pt in points3D.items()))/len(points3D)
    HEADER = "# 3D point list with one line of data per point:\n" + \
             "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n" + \
             "# Number of points: {}, mean track length: {}\n".format(len(points3D), mean_track_length)

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, pt in points3D.items():
            point_header = [pt.id, *pt.xyz, *pt.rgb, pt.error]
            fid.write(" ".join(map(str, point_header)) + " ")
            track_strings = []
            for image_id, point2D in zip(pt.image_ids, pt.point2D_idxs):
                track_strings.append(" ".join(map(str, [image_id, point2D])))
            fid.write(" ".join(track_strings) + "\n")
            

def write_points3D_binary(points3D, path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(points3D), "Q")
        for _, pt in points3D.items():
            write_next_bytes(fid, pt.id, "Q")
            write_next_bytes(fid, pt.xyz.tolist(), "ddd")
            write_next_bytes(fid, pt.rgb.tolist(), "BBB")
            write_next_bytes(fid, pt.error, "d")
            track_length = pt.image_ids.shape[0]
            write_next_bytes(fid, track_length, "Q")
            for image_id, point2D_id in zip(pt.image_ids, pt.point2D_idxs):
                write_next_bytes(fid, [image_id, point2D_id], "ii")


def quaternion_to_rotation_matrix(qvec):
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec
    R = np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]])
    return R


def camera_center_to_translation(c, qvec):
    R = quaternion_to_rotation_matrix(qvec)
    return (-1) * np.matmul(R, c)



def read_nvm_model(nvm_path, width=1920, height=1080, skip_points=False):
    logging.info(f'Camera height={height} and width={width}')

    nvm_f = open(nvm_path, 'r')
    line = nvm_f.readline()
    while line == '\n' or line.startswith('NVM_V3'):
        line = nvm_f.readline()
    num_images = int(line)


    logging.info(f'Reading {num_images} images...')
    image_data = []
    i = 0
    while i < num_images:
        line = nvm_f.readline()
        if line == '\n':
            continue
        data = line.strip('\n').split()
        image_data.append(data)
        i += 1

    line = nvm_f.readline()
    while line == '\n':
        line = nvm_f.readline()
    num_points = int(line)

    if skip_points:
        logging.info(f'Skipping {num_points} points.')
        num_points = 0
    else:
        logging.info(f'Reading {num_points} points...')
    points3D = {}
    image_idx_to_keypoints = defaultdict(list)
    i = 0
    pbar = tqdm(total=num_points, unit='pts')
    while i < num_points:
        line = nvm_f.readline()
        if line == '\n':
            continue

        data = line.strip('\n').split(' ')
        x, y, z, r, g, b, num_observations = data[:7]
        obs_image_ids, point2D_idxs = [], []
        for j in range(int(num_observations)):
            s = 7 + 4*j
            img_index, kp_index, kx, ky = data[s:s+4]
            image_idx_to_keypoints[int(img_index)].append(
                (int(kp_index), float(kx), float(ky), i))
            db_image_id = int(img_index)
            obs_image_ids.append(db_image_id)
            point2D_idxs.append(kp_index)

        point = Point3D(
            id=i,
            xyz=np.array([x, y, z], float),
            rgb=np.array([r, g, b], int),
            error=1.,  # fake
            image_ids=np.array(obs_image_ids, int),
            point2D_idxs=np.array(point2D_idxs, int))
        points3D[i] = point

        i += 1
        pbar.update(1)
    pbar.close()

    logging.info('Parsing image data...')
    images, cameras = {}, {}
    for idx, data in enumerate(image_data):
        name, f, qw, qx, qy, qz, cx, cy, cz, k, _ = data
        name = name.replace('.jpg', '.png')
        qvec = np.array([qw, qx, qy, qz], float)
        c = np.array([cx, cy, cz], float)
        t = camera_center_to_translation(c, qvec)
        if t.max() > 1e4:
            print("skip outlier", name)
            continue
            

        if i in image_idx_to_keypoints:
            # NVM only stores triangulated 2D keypoints: add dummy ones
            keypoints = image_idx_to_keypoints[i]
            point2D_idxs = np.array([d[0] for d in keypoints])
            tri_xys = np.array([[x, y] for _, x, y, _ in keypoints])
            tri_ids = np.array([i for _, _, _, i in keypoints])

            num_2Dpoints = max(point2D_idxs) + 1
            xys = np.zeros((num_2Dpoints, 2), float)
            point3D_ids = np.full(num_2Dpoints, -1, int)
            xys[point2D_idxs] = tri_xys
            point3D_ids[point2D_idxs] = tri_ids
        else:
            xys = np.zeros((0, 2), float)
            point3D_ids = np.full(0, -1, int)

        image_id = int(idx)
        image = Image(
            id=image_id,
            qvec=qvec,
            tvec=t,
            camera_id=int(idx),
            name=name,
            xys=xys,
            point3D_ids=point3D_ids)
        images[image_id] = image

        camera_model = CAMERA_MODEL_NAMES['SIMPLE_RADIAL']
        px, py = width / 2., height /2.
        params = [float(f), px, py, -float(k)] # Take care of the difference between colmap and visual sfm
        camera = Camera(
            id=int(idx), model=camera_model.model_name,
            width=int(width), height=int(height), params=params)
        cameras[int(idx)] = camera

    return cameras, images, points3D


def convert_nvm_to_colmap(nvm_path, colmap_path, image_width, image_height, 
                          skip_point=False, save_txt=False):
    
    cameras, images, points3D = \
        read_nvm_model(nvm_path, image_width, image_height, skip_point)
    
    logging.info(f'Cameara num={len(cameras)}, image num={len(images)}, points num={len(points3D)}')
    
    os.makedirs(colmap_path, exist_ok=True)
    
    if save_txt:
        logging.info(f'Save txt format at {colmap_path}')
        write_cameras_text(cameras, os.path.join(colmap_path, 'cameras.txt'))
        write_images_text(images, os.path.join(colmap_path, 'images.txt'))
        write_points3D_text(points3D, os.path.join(colmap_path, 'points3D.txt'))
    else:
        logging.info(f'Save binary format at {colmap_path}')
        write_cameras_binary(cameras, os.path.join(colmap_path, 'cameras.bin'))
        write_images_binary(images, os.path.join(colmap_path, 'images.bin'))
        write_points3D_binary(points3D, os.path.join(colmap_path, 'points3D.bin'))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nvm_path', required=True, type=str)
    parser.add_argument('--colmap_path', required=True, type=str)
    parser.add_argument('--image_width', required=False, default=1920, type=int)
    parser.add_argument('--image_height', required=False, default=1080, type=int)
    parser.add_argument('--skip_point', action='store_true')
    parser.add_argument('--save_txt', action='store_true')
    args = parser.parse_args()
    convert_nvm_to_colmap(**args.__dict__)