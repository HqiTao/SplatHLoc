import torch
# from torch_scatter import scatter_max

def farthest_point_sample(data,npoints):
    """
    Args:
        data:输入的tensor张量，排列顺序 N,D
        Npoints: 需要的采样点

    Returns:data->采样点集组成的tensor，每行是一个采样点
    """
    N,D = data.shape #N是点数，D是维度
    xyz = data[:,:3] #只需要坐标
    centroids = torch.zeros(size=(npoints,)) #最终的采样点index
    dictance = torch.ones(size=(N,))*1e10 #距离列表,一开始设置的足够大,保证第一轮肯定能更新dictance
    farthest = torch.ones(size=(1,)) #随机选一个采样点的index
    for i in range(npoints):
        centroids[i] = farthest
        centroid = xyz[farthest,:]
        dict = ((xyz-centroid)**2).sum(dim=-1)
        mask = dict < dictance
        dictance[mask] = dict[mask]
        farthest = torch.argmax(dictance,dim=-1)
    print(centroids.type(torch.long))
    data= data[centroids.type(torch.long)]
    return data



@torch.no_grad()
def kde(x, std = 0.1, half = True, down = None):
    # use a gaussian kernel to estimate density
    if half:
        x = x.half() # Do it in half precision TODO: remove hardcoding
    if down is not None:
        scores = (-torch.cdist(x,x[::down])**2/(2*std**2)).exp()
    else:
        scores = (-torch.cdist(x,x)**2/(2*std**2)).exp()
    density = scores.sum(dim=-1)
    return density

@torch.no_grad()
def s_fps(data, score, npoints):
    """
    Args:
        data:输入的tensor张量，排列顺序 N,D
        Npoints: 需要的采样点

    Returns:data->采样点集组成的tensor，每行是一个采样点
    """
    
    N,D = data.shape #N是点数，D是维度
    xyz = data[:,:3] #只需要坐标
    centroids = torch.zeros(size=(npoints,), device=xyz.device) #最终的采样点index
    dictance = torch.ones(size=(N,), device=xyz.device)*1e10 #距离列表,一开始设置的足够大,保证第一轮肯定能更新dictance
    farthest = torch.ones(size=(1,), device=xyz.device).int() #随机选一个采样点的index
    if score is None:
        score = torch.ones(size=(N,), device=xyz.device)

    for i in range(npoints):
        centroids[i] = farthest
        centroid = xyz[farthest,:]
        dict = ((xyz-centroid)**2).sum(dim=-1)
        mask = dict < dictance
        dictance[mask] = dict[mask]
        weighted_dictance = dictance * score
        farthest = torch.argmax(weighted_dictance, dim=-1)
    print(centroids.type(torch.long))
    data= data[centroids.type(torch.long)]
    return centroids.type(torch.long)

# def weighted_voxel_sample(xyz, weights, voxel_size):
#     """
#     Args:
#         xyz: 输入的 tensor 张量，排列顺序 N,D
#         weights: 权重
#         voxel_size: voxel 的大小

#     Returns:
#         data: 采样点集组成的 tensor，每行是一个采样点
#     """
#     xyz_min = xyz.min(dim=0)[0]
#     xyz_max = xyz.max(dim=0)[0]
#     xyz = xyz - xyz_min
#     voxel_size = torch.tensor(voxel_size).type_as(xyz)
#     voxel_grid_size = torch.ceil((xyz_max - xyz_min) / voxel_size).long()
#     voxel_grid_size = torch.max(voxel_grid_size, torch.ones_like(voxel_grid_size))
#     voxel_num = voxel_grid_size[0] * voxel_grid_size[1] * voxel_grid_size[2]

#     # 每个点的 voxel 索引
#     voxel_idx = (xyz / voxel_size).long()
#     voxel_idx_flat = voxel_idx[:, 0] + voxel_idx[:, 1] * voxel_grid_size[0] + voxel_idx[:, 2] * voxel_grid_size[0] * voxel_grid_size[1]

#     # 相同 voxel 内的点取 weights 最大的点
    

#     out, argmax = scatter_max(weights, voxel_idx_flat, dim=0)

#     argmax_idx_valid = argmax < weights.shape[0]

#     voxel_counts = torch.bincount(voxel_idx_flat, minlength=voxel_num)
#     print("voxel_counts", voxel_counts.max())
    
#     sample_mask = torch.zeros(weights.shape[0], dtype=torch.bool, device=weights.device)
#     sample_mask[argmax[argmax_idx_valid]] = True  # 确保只有有效索引被标记为 True

#     return sample_mask




    