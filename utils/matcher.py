import torch
import torch.nn.functional as F
from lightglue.utils import load_image, rbd

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


@torch.no_grad()
def sparse_matcher(image0, image1, extractor, matcher, data):
    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)

    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
    matches = matches01['matches']  # indices with shape (K,2)
    points0 = feats0['keypoints'][matches[..., 0]].cpu().numpy()
    points1 = feats1['keypoints'][matches[..., 1]].cpu().numpy()

    query_p2d  = torch.from_numpy(points0).float().to("cuda")
    rendered_p2d = torch.from_numpy(points1).float().to("cuda")

    data.update({
        'mkpts0_f': query_p2d,
        'mkpts1_f': rendered_p2d
    })

    return data


@torch.no_grad()
def hybrid_matcher_f(coarse_query_feature_map, coarse_rendered_feature_map, C, temp, thr, fine_matcher, data):
    # coarse match
    coarse_corr_matrix = torch.matmul(
        coarse_query_feature_map.permute(1, 2, 0).reshape(1, -1, C),
        coarse_rendered_feature_map.reshape(1, C, -1),
    )  # 1, N, M

    coarse_corr_matrix = dual_softmax(
        coarse_corr_matrix, temp=temp
    )

    c_b_ids, c_i_ids, c_j_ids = mnn_match(
        coarse_corr_matrix, thr=thr
    )

    if c_i_ids.dim() == 0:
        print("[skip] Failed in coarse match")
        return None
    elif c_i_ids.shape[0] < 10:
        print("[skip] Failed in coarse match")
        return None
    
    # fine match
    with torch.autocast(enabled=False, device_type='cuda'):
        fine_matcher.backbone(data)
        data.update({
            'hw0_i': data['imagec_0'].shape[2:],
            'hw1_i': data['imagec_1'].shape[2:],
            'hw0_c': [data['h_8'], data['w_8']],
            'hw1_c': [data['h_8'], data['w_8']],
        })
        fine_matcher.matcher.coarse_match(data)

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
        feat_f0_unfold, feat_f1_unfold = fine_matcher.matcher.fine_preprocess(data, None)

        fine_matcher.matcher.fine_matching(feat_f0_unfold.transpose(1, 2), feat_f1_unfold.transpose(1, 2), data)

        return data


@torch.no_grad()
def hybrid_matcher_c(coarse_query_feature_map, C, temp, thr, fine_query_feature_map, fine_rendered_feature_map, sd_matcher, data):
    # coarse match
    with torch.autocast(enabled=False, device_type='cuda'):
        sd_matcher.backbone(data)
        data.update({
            'hw0_i': data['imagec_0'].shape[2:],
            'hw1_i': data['imagec_1'].shape[2:],
            'hw0_c': [data['h_8'], data['w_8']],
            'hw1_c': [data['h_8'], data['w_8']],
        })
        sd_matcher.matcher.coarse_match(data)

    c_b_ids = data['b_ids']
    c_i_ids = data['i_ids']
    c_j_ids = data['j_ids']
 

    if c_i_ids.dim() == 0:
        print("[skip] Failed in coarse match")
        return None
    elif c_i_ids.shape[0] < 10:
        print("[skip] Failed in coarse match")
        return None
    
    # fine match
    Hf, Wf = fine_query_feature_map.shape[-2:]
    Hc, Wc = coarse_query_feature_map.shape[-2:]
    W = Hf // Hc  # window size
    assert W == 8
    overlap_size = 0  
    WW = W * W

    query_feature_windows = (
        F.unfold(
            fine_query_feature_map, (W, W), stride=W, padding=overlap_size // 2
        )
        .reshape(1, C, WW, -1)[c_b_ids, :, :, c_i_ids]
        .permute(0, 2, 1)
    )  # B, N, C
    rendered_feature_windows = (
        F.unfold(
            fine_rendered_feature_map, (W, W), stride=W, padding=overlap_size // 2
        )
        .reshape(1, C, WW, -1)[c_b_ids, :, :, c_j_ids]
        .permute(0, 2, 1)
    )  # B, M, C

    fine_corr_matrix = torch.matmul(
        query_feature_windows, rendered_feature_windows.transpose(-2, -1)
    )  # B, N, M

    fine_corr_matrix = dual_softmax(
        fine_corr_matrix, temp=temp
    )

    f_b_ids, f_i_ids, f_j_ids = mnn_match(
        fine_corr_matrix, thr=thr
    )

    if f_i_ids.dim() == 0:
        print("[skip] Failed in fine match")
        return None
    elif f_i_ids.shape[0] < 3:
        print("[skip] Failed in fine match")
        return None

    query_p2d = torch.stack(
        [
            c_i_ids[f_b_ids] % Wc * W + f_i_ids % W,
            c_i_ids[f_b_ids] // Wc * W + f_i_ids // W,
        ],
        dim=1,
    ).float()
    rendered_p2d = torch.stack(
        [
            c_j_ids[f_b_ids] % Wc * W + f_j_ids % W,
            c_j_ids[f_b_ids] // Wc * W + f_j_ids // W,
        ],
        dim=1,
    ).float()

    data.update({
        'mkpts0_f': query_p2d,
        'mkpts1_f': rendered_p2d
    })

    return data


@torch.no_grad()
def fgs_matcher(coarse_query_feature_map, coarse_rendered_feature_map, C, temp, thr, fine_query_feature_map, fine_rendered_feature_map, data):
    Hf, Wf = fine_query_feature_map.shape[-2:]
    Hc, Wc = coarse_query_feature_map.shape[-2:]
    W = Hf // Hc  # window size
    assert W == 8
    overlap_size = 0  
    WW = W * W

    # coarse match
    coarse_corr_matrix = torch.matmul(
        coarse_query_feature_map.permute(1, 2, 0).reshape(1, -1, C),
        coarse_rendered_feature_map.reshape(1, C, -1),
    )  # 1, N, M

    coarse_corr_matrix = dual_softmax(
        coarse_corr_matrix, temp=temp
    )

    c_b_ids, c_i_ids, c_j_ids = mnn_match(
        coarse_corr_matrix, thr=thr
    )

    if c_i_ids.dim() == 0:
        print("[skip] Failed in coarse match")
        return None
    elif c_i_ids.shape[0] < 3:
        print("[skip] Failed in coarse match")
        return None
    
    # fine match
    query_feature_windows = (
        F.unfold(
            fine_query_feature_map, (W, W), stride=W, padding=overlap_size // 2
        )
        .reshape(1, C, WW, -1)[c_b_ids, :, :, c_i_ids]
        .permute(0, 2, 1)
    )  # B, N, C
    rendered_feature_windows = (
        F.unfold(
            fine_rendered_feature_map, (W, W), stride=W, padding=overlap_size // 2
        )
        .reshape(1, C, WW, -1)[c_b_ids, :, :, c_j_ids]
        .permute(0, 2, 1)
    )  # B, M, C

    fine_corr_matrix = torch.matmul(
        query_feature_windows, rendered_feature_windows.transpose(-2, -1)
    )  # B, N, M

    fine_corr_matrix = dual_softmax(
        fine_corr_matrix, temp=temp
    )

    f_b_ids, f_i_ids, f_j_ids = mnn_match(
        fine_corr_matrix, thr=thr
    )

    if f_i_ids.dim() == 0:
        print("[skip] Failed in fine match")
        return None
    elif f_i_ids.shape[0] < 3:
        print("[skip] Failed in fine match")
        return None

    query_p2d = torch.stack(
        [
            c_i_ids[f_b_ids] % Wc * W + f_i_ids % W,
            c_i_ids[f_b_ids] // Wc * W + f_i_ids // W,
        ],
        dim=1,
    ).float()
    rendered_p2d = torch.stack(
        [
            c_j_ids[f_b_ids] % Wc * W + f_j_ids % W,
            c_j_ids[f_b_ids] // Wc * W + f_j_ids // W,
        ],
        dim=1,
    ).float()

    data.update({
        'mkpts0_f': query_p2d,
        'mkpts1_f': rendered_p2d
    })

    return data


@torch.no_grad()
def semi_dense_matcher(sd_matcher, data):
    with torch.autocast(enabled=False, device_type='cuda'):
        sd_matcher.backbone(data)
        data.update({
            'hw0_i': data['imagec_0'].shape[2:],
            'hw1_i': data['imagec_1'].shape[2:],
            'hw0_c': [data['h_8'], data['w_8']],
            'hw1_c': [data['h_8'], data['w_8']],
        })
        sd_matcher.matcher.coarse_match(data)

        feat_f0_unfold, feat_f1_unfold = sd_matcher.matcher.fine_preprocess(data, None)

        sd_matcher.matcher.fine_matching(feat_f0_unfold.transpose(1, 2), feat_f1_unfold.transpose(1, 2), data)

    return data
