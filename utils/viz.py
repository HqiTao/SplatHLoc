import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import torch
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

def cm_RdGn(x):
    """Custom colormap: red (0) -> yellow (0.5) -> green (1)."""
    x = np.clip(x, 0, 1)[..., None]*2
    c = x*np.array([[0, 1., 0]]) + (2-x)*np.array([[1., 0, 0]])
    return np.clip(c, 0, 1)


def plot_images(imgs, titles=None, cmaps='gray', dpi=100, pad=.5,
                adaptive=True):
    """Plot a set of images horizontally.
    Args:
        imgs: a list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        adaptive: whether the figure size should fit the image aspect ratios.
    """
    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    if adaptive:
        ratios = [i.shape[1] / i.shape[0] for i in imgs]  # W / H
    else:
        ratios = [4/3] * n
    figsize = [sum(ratios)*4.5, 4.5]
    fig, ax = plt.subplots(
        1, n, figsize=figsize, dpi=dpi, gridspec_kw={'width_ratios': ratios})
    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i])
    fig.tight_layout(pad=pad)


def plot_keypoints(kpts, colors='lime', ps=4):
    """Plot keypoints for existing images.
    Args:
        kpts: list of ndarrays of size (N, 2).
        colors: string, or list of list of tuples (one for each keypoints).
        ps: size of the keypoints as float.
    """
    if not isinstance(colors, list):
        colors = [colors] * len(kpts)
    axes = plt.gcf().axes
    for a, k, c in zip(axes, kpts, colors):
        a.scatter(k[:, 0], k[:, 1], c=c, s=ps, linewidths=0)


def plot_matches(kpts0, kpts1, color=None, scores=None, lw=1.5, ps=4, indices=(0, 1), a=1.):
    """Plot matches for a pair of existing images.
    Args:
        kpts0, kpts1: corresponding keypoints of size (N, 2).
        color: color of each match, string or RGB tuple. Random if not given.
        scores: score of each match
        lw: width of the lines.
        ps: size of the end points (no endpoint if ps=0)
        indices: indices of the images to draw the matches on.
        a: alpha opacity of the match lines.
    """
    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    ax0, ax1 = ax[indices[0]], ax[indices[1]]
    fig.canvas.draw()

    assert len(kpts0) == len(kpts1)
    if color is None:
        if scores is None:
            color = matplotlib.cm.hsv(np.random.rand(len(kpts0))).tolist()
        else:
            color = [list(*cm_RdGn(x)) for x in scores]
    elif len(color) > 0 and not isinstance(color[0], (tuple, list)):
        color = [color] * len(kpts0)

    if lw > 0:
        # transform the points into the figure coordinate system
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(ax0.transData.transform(kpts0))
        fkpts1 = transFigure.transform(ax1.transData.transform(kpts1))
        fig.lines += [matplotlib.lines.Line2D(
            (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]),
            zorder=1, transform=fig.transFigure, c=color[i], linewidth=lw,
            alpha=a)
            for i in range(len(kpts0))]

    # freeze the axes to prevent the transform to change
    ax0.autoscale(enable=False)
    ax1.autoscale(enable=False)

    # print(len(color), color[0])
    if ps > 0:
        ax0.scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
        ax1.scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def plot_matches_w_gt_point(kpts0, kpts1, kpts0_gt_in_2, color=None, scores=None, lw=1.5, ps=4, indices=(0, 1), a=1.):
    """Plot matches for a pair of existing images.
    Args:
        kpts0, kpts1: corresponding keypoints of size (N, 2).
        color: color of each match, string or RGB tuple. Random if not given.
        scores: score of each match
        lw: width of the lines.
        ps: size of the end points (no endpoint if ps=0)
        indices: indices of the images to draw the matches on.
        a: alpha opacity of the match lines.
    """
    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    ax0, ax1 = ax[indices[0]], ax[indices[1]]
    fig.canvas.draw()

    assert len(kpts0) == len(kpts1)
    if color is None:
        if scores is None:
            color = matplotlib.cm.hsv(np.random.rand(len(kpts0))).tolist()
        else:
            color = [list(*cm_RdGn(x)) for x in scores]
    elif len(color) > 0 and not isinstance(color[0], (tuple, list)):
        color = [color] * len(kpts0)

    if lw > 0:
        # transform the points into the figure coordinate system
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(ax0.transData.transform(kpts0))
        fkpts1 = transFigure.transform(ax1.transData.transform(kpts1))
        
        fig.lines += [matplotlib.lines.Line2D(
            (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]),
            zorder=1, transform=fig.transFigure, c=color[i], linewidth=lw,
            alpha=a)
            for i in range(len(kpts0))]

    # freeze the axes to prevent the transform to change
    ax0.autoscale(enable=False)
    ax1.autoscale(enable=False)

    # print(len(color), color[0])
    if ps > 0:
        ax0.scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
        ax1.scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def add_text(idx, text, pos=(0.01, 0.99), fs=15, color='w',
             lcolor='k', lwidth=2, ha='left', va='top'):
    ax = plt.gcf().axes[idx]
    t = ax.text(*pos, text, fontsize=fs, ha=ha, va=va,
                color=color, transform=ax.transAxes)
    if lcolor is not None:
        t.set_path_effects([
            path_effects.Stroke(linewidth=lwidth, foreground=lcolor),
            path_effects.Normal()])


def save_plot(path, **kw):
    """Save the current figure without any white margin."""
    plt.savefig(path, bbox_inches='tight', pad_inches=0, **kw)
    
    
def draw_keypoint(image, kpts, scores=None, ps=4):
    """Draw keypoint on image
    Args:
        image: RGB, HWC
        kps: N*2
        color: RGB
    """
    plot_images([image])
    
    if scores is None:
        colors='lime'
    else:
        colors = [list(*cm_RdGn(x)) for x in scores]
    plot_keypoints([kpts], colors, ps=ps)
    

def draw_matches(image0, image1, kpts0, kpts1, scores, lw=1.5, 
                 ps=4, alpha=1.0):
    """Draw keypoint on image
    Args:
        image: RGB, HWC
        kps: N*2
        color: RGB
    """
    plot_images([image0, image1])
    colors = [list(*cm_RdGn(x)) for x in scores]
    plot_matches(kpts0, kpts1, colors, lw=lw, ps=ps, a=alpha)



def slerp(r1, r2, t):
    """Spherical interpolation between two rotations.
    Args:
        r1, r2: 3x3 rotation matrices.
        t: interpolation factor.
    Returns:
        3x3 interpolated rotation matrix.
    """
    # r1 = R.from_matrix(r1)
    # r2 = R.from_matrix(r2)
    times = []
    for i in range(t):
        times.append(i)
    key_times = [times[0], times[-1]]
    slerp = Slerp(key_times, R.from_matrix([r1, r2]))
    inter_r = slerp(times)
    return inter_r


def interpolate_pose(pose1, pose2, t):
    """Interpolate between two poses.
    Args:
        pose1, pose2: 4x4 transformation matrices.
        t: interpolation factor.
    Returns:
        4x4 interpolated transformation matrix.
    """
    device = pose1.device
    pose1 = pose1.cpu().numpy()
    pose2 = pose2.cpu().numpy()
    r1, r2 = pose1[:3, :3], pose2[:3, :3]
    t1, t2 = pose1[:3, 3], pose2[:3, 3]

    # interpolate rotation
    rs = slerp(r1, r2, t)

    # interpolate translation
    ts = []
    for t in np.linspace(0, 1, t):
        position = (1 - t) * t1 + t * t2
        ts.append(position)

    # compose the interpolated pose
    poses = []
    for r_t, t_t in zip(rs, ts):
        inter_pose = np.eye(4)
        inter_pose[:3, :3] = r_t.as_matrix()
        inter_pose[:3, 3] = t_t
        poses.append(torch.tensor(inter_pose, device=device, dtype=torch.float32))

    return poses

def stitch_images(image_a, image_b, mask_left=None, mask_right=None):
    if mask_left is None:
        print(image_a.shape)
        width, height = image_a.shape[:2]
        mask_left = np.zeros((width, height), dtype=bool)

        a = width/height
        for x in range(width):
            for y in range(height):
                if x > y*a:  
                    mask_left[x, y] = True
                else:  
                    continue
        mask_right = ~mask_left
    
    stitched_image = np.zeros_like(image_a)
    stitched_image[mask_left] = image_a[mask_left]
    stitched_image[mask_right] = image_b[mask_right]
    return stitched_image, mask_left, mask_right
    
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import numpy as np
from sklearn.decomposition import PCA
import torch.nn.functional as F
from collections import defaultdict, deque
import torch
import torch.nn as nn

import os
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont


class RollingAvg:

    def __init__(self, length):
        self.length = length
        self.metrics = defaultdict(lambda: deque(maxlen=self.length))

    def add(self, name, metric):
        self.metrics[name].append(metric)

    def get(self, name):
        return torch.tensor(list(self.metrics[name])).mean()

    def logall(self, log_func):
        for k in self.metrics.keys():
            log_func(k, self.get(k))


def _remove_axes(ax):
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.set_xticks([])
    ax.set_yticks([])


def remove_axes(axes):
    if len(axes.shape) == 2:
        for ax1 in axes:
            for ax in ax1:
                _remove_axes(ax)
    else:
        for ax in axes:
            _remove_axes(ax)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        if len(image2.shape) == 4:
            # batched
            image2 = image2.permute(1, 0, 2, 3)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2.permute(1, 0, 2, 3)


norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

midas_norm = T.Normalize([0.5] * 3, [0.5] * 3)
midas_unnorm = UnNormalize([0.5] * 3, [0.5] * 3)


class ToTargetTensor(object):
    def __call__(self, target):
        return torch.as_tensor(np.array(target), dtype=torch.int64).unsqueeze(0)


def show_heatmap(ax,
                 image,
                 heatmap,
                 cmap="bwr",
                 color=False,
                 center=False,
                 show_negative=False,
                 cax=None,
                 vmax=None):
    frame = []

    if color:
        frame.append(ax.imshow(image))
    else:
        bw = np.dot(np.array(image)[..., :3] / 255, [0.2989, 0.5870, 0.1140])
        bw = np.ones_like(image) * np.expand_dims(bw, -1)
        frame.append(ax.imshow(bw))

    if center:
        heatmap -= heatmap.mean()

    if not show_negative:
        heatmap = heatmap.clamp_min(0)

    heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), (image.shape[0], image.shape[1])) \
        .squeeze(0).squeeze(0)

    if vmax is None:
        vmax = np.abs(heatmap).max()

    hm = ax.imshow(heatmap, alpha=.5, cmap=cmap, vmax=vmax, vmin=-vmax)
    if cax is not None:
        plt.colorbar(hm, cax=cax, orientation='vertical')

    frame.extend([hm])
    return frame


def implicit_feats(original_image, input_size, color_feats):
    n_freqs = 20
    grid = torch.linspace(-1, 1, input_size, device=original_image.device)
    feats = torch.cat([t.unsqueeze(0) for t in torch.meshgrid([grid, grid])]).unsqueeze(0)

    if color_feats:
        feat_list = [feats, original_image]
        dim_multiplier = 5
    else:
        feat_list = [feats]
        dim_multiplier = 2

    feats = torch.cat(feat_list, dim=1)
    freqs = torch.exp(torch.linspace(-2, 10, n_freqs, device=original_image.device)) \
        .reshape(n_freqs, 1, 1, 1)
    feats = (feats * freqs).reshape(1, n_freqs * dim_multiplier, input_size, input_size)

    if color_feats:
        all_feats = [torch.sin(feats), torch.cos(feats), original_image]
    else:
        all_feats = [torch.sin(feats), torch.cos(feats)]
    return torch.cat(all_feats, dim=1)


def load_hr_emb(original_image, model_path, color_feats=True):
    model = torch.load(model_path, map_location="cpu")
    hr_model = model["model"].cuda().eval()
    unprojector = model["unprojector"].cuda().eval()

    with torch.no_grad():
        h, w = original_image.shape[2:]
        assert h == w
        feats = implicit_feats(original_image, h, color_feats).cuda()
        hr_feats = hr_model(feats)
        hr_feats = unprojector(hr_feats.detach().cpu())

        return hr_feats


def generate_subset(n, batch):
    np.random.seed(0)
    return np.random.permutation(n)[:batch]


class TorchPCA(object):

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(dim=0)
        unbiased = X - self.mean_.unsqueeze(0)
        U, S, V = torch.pca_lowrank(unbiased, q=self.n_components, center=False, niter=4)
        self.components_ = V.T
        self.singular_values_ = S
        return self

    def transform(self, X):
        t0 = X - self.mean_.unsqueeze(0)
        projected = t0 @ self.components_.T
        return projected


def pca(image_feats_list, dim=3, fit_pca=None, use_torch_pca=True, max_samples=None):
    device = image_feats_list[0].device

    def flatten(tensor, target_size=None):
        if target_size is not None and fit_pca is None:
            tensor = F.interpolate(tensor, (target_size, target_size), mode="bilinear")
        B, C, H, W = tensor.shape
        return tensor.permute(1, 0, 2, 3).reshape(C, B * H * W).permute(1, 0).detach().cpu()

    if len(image_feats_list) > 1 and fit_pca is None:
        target_size = image_feats_list[0].shape[2]
    else:
        target_size = None

    flattened_feats = []
    for feats in image_feats_list:
        flattened_feats.append(flatten(feats, target_size))
    x = torch.cat(flattened_feats, dim=0)

    # Subsample the data if max_samples is set and the number of samples exceeds max_samples
    if max_samples is not None and x.shape[0] > max_samples:
        indices = torch.randperm(x.shape[0])[:max_samples]
        x = x[indices]

    if fit_pca is None:
        if use_torch_pca:
            fit_pca = TorchPCA(n_components=dim).fit(x)
        else:
            fit_pca = PCA(n_components=dim).fit(x)

    reduced_feats = []
    for feats in image_feats_list:
        x_red = fit_pca.transform(flatten(feats))
        if isinstance(x_red, np.ndarray):
            x_red = torch.from_numpy(x_red)
        x_red -= x_red.min(dim=0, keepdim=True).values
        x_red /= x_red.max(dim=0, keepdim=True).values
        B, C, H, W = feats.shape
        reduced_feats.append(x_red.reshape(B, H, W, dim).permute(0, 3, 1, 2).to(device))

    return reduced_feats, fit_pca


class PCAUnprojector(nn.Module):

    def __init__(self, feats, dim, device, use_torch_pca=False, **kwargs):
        super().__init__()
        self.dim = dim

        if feats is not None:
            self.original_dim = feats.shape[1]
        else:
            self.original_dim = kwargs["original_dim"]

        if self.dim != self.original_dim:
            if feats is not None:
                sklearn_pca = pca([feats], dim=dim, use_torch_pca=use_torch_pca)[1]

                # Register tensors as buffers
                self.register_buffer('components_',
                                     torch.tensor(sklearn_pca.components_, device=device, dtype=feats.dtype))
                self.register_buffer('singular_values_',
                                     torch.tensor(sklearn_pca.singular_values_, device=device, dtype=feats.dtype))
                self.register_buffer('mean_', torch.tensor(sklearn_pca.mean_, device=device, dtype=feats.dtype))
            else:
                self.register_buffer('components_', kwargs["components_"].t())
                self.register_buffer('singular_values_', kwargs["singular_values_"])
                self.register_buffer('mean_', kwargs["mean_"])

        else:
            print("PCAUnprojector will not transform data")

    def forward(self, red_feats):
        if self.dim == self.original_dim:
            return red_feats
        else:
            b, c, h, w = red_feats.shape
            red_feats_reshaped = red_feats.permute(0, 2, 3, 1).reshape(b * h * w, c)
            unprojected = (red_feats_reshaped @ self.components_) + self.mean_.unsqueeze(0)
            return unprojected.reshape(b, h, w, self.original_dim).permute(0, 3, 1, 2)

    def project(self, feats):
        if self.dim == self.original_dim:
            return feats
        else:
            b, c, h, w = feats.shape
            feats_reshaped = feats.permute(0, 2, 3, 1).reshape(b * h * w, c)
            t0 = feats_reshaped - self.mean_.unsqueeze(0).to(feats.device)
            projected = t0 @ self.components_.t().to(feats.device)
            return projected.reshape(b, h, w, self.dim).permute(0, 3, 1, 2)


def prep_image(t, subtract_min=True):
    if subtract_min:
        t -= t.min()
    t /= t.max()
    t = (t * 255).clamp(0, 255).to(torch.uint8)

    if len(t.shape) == 2:
        t = t.unsqueeze(0)

    return t

def save_render_comparison(
    query_img,
    render_sparse,
    render_dense,
    angle_deg,
    sparse_ae, sparse_te,
    dense_ae, dense_te,
    save_path="output/comparison.png",
    top_margin=40,
    bottom_margin=20,
    side_margin=20,
    middle_margin=20
):
    """
    拼接 query + sparse + dense 三张图，并添加 AE/TE 注释，四周留空白。

    参数：
        query_img: torch.Tensor, [3, H, W]，原始查询图像
        render_sparse: torch.Tensor, sparse 渲染结果
        render_dense: torch.Tensor, dense 渲染结果
        sparse_ae, sparse_te: sparse 精度指标
        dense_ae, dense_te: dense 精度指标
        save_path: 保存路径
        top_margin: 图像顶部空白高度
        bottom_margin: 图像底部空白高度
        side_margin: 左右两侧空白宽度
        middle_margin: 图像之间的间隔宽度
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 转为 PIL 图像
    def tensor_to_pil(t):
        t = t.detach().cpu().clamp(0, 1)
        if t.dim() == 4 and t.size(0) == 1:
            t = t.squeeze(0)  # 去掉 batch 维度
        return TF.to_pil_image(t)

    img_query = tensor_to_pil(query_img)
    img_sparse = tensor_to_pil(render_sparse)
    img_dense = tensor_to_pil(render_dense)

    # 获取尺寸
    w, h = img_query.width, img_query.height
    total_width = w * 3 + middle_margin * 2 + side_margin * 2
    total_height = h + top_margin + bottom_margin

    # 创建画布
    canvas = Image.new("RGB", (total_width, total_height), (255, 255, 255))

    # 粘贴图像：从左到右依次为 query, sparse, dense
    x_query = side_margin
    x_sparse = x_query + w + middle_margin
    x_dense = x_sparse + w + middle_margin
    y = top_margin

    canvas.paste(img_query, (x_query, y))
    canvas.paste(img_sparse, (x_sparse, y))
    canvas.paste(img_dense, (x_dense, y))

    # 添加文字
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=18)
    except:
        font = ImageFont.load_default()

    draw.text((x_query + 10, 10), f"[Query] AE: {angle_deg:.2f}°", fill=(0, 0, 0), font=font)
    draw.text((x_sparse + 10, 10), f"[Sparse] AE: {sparse_ae:.2f}°, TE: {sparse_te:.2f}cm", fill=(255, 0, 0), font=font)
    draw.text((x_dense + 10, 10), f"[VGGT] AE: {dense_ae:.2f}°, TE: {dense_te:.2f}cm", fill=(0, 0, 255), font=font)

    # 自动添加扩展名
    if not save_path.lower().endswith((".png", ".jpg", ".jpeg")):
        save_path += ".png"

    # 保存图像
    canvas.save(save_path)