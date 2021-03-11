from typing import List, Tuple, Optional

import numpy as np
import open3d as o3d


def roi_rectangle(pcdarray: np.ndarray, minxy: np.ndarray, size: np.ndarray) -> np.ndarray:
    maxxy = minxy + size
    xy_pts = pcdarray[:, [0, 1]]
    inidx = np.all((minxy <= xy_pts) & (xy_pts <= maxxy), axis=1)
    return pcdarray[inidx]


def get_rectangle_pcd(pcdarray: np.ndarray, minxy: np.ndarray, size: np.ndarray) -> Optional[o3d.geometry.PointCloud]:
    pts = roi_rectangle(pcdarray, minxy, size)
    if len(pts) == 0:
        return None
    pcd_voxel = o3d.geometry.PointCloud()
    pcd_voxel.points = o3d.utility.Vector3dVector(pts)
    return pcd_voxel


def compute_overlap_ratio(pcd0: o3d.geometry.PointCloud, pcd1: o3d.geometry.PointCloud, voxel_size: float):
    pcd0_down = pcd0.voxel_down_sample(voxel_size)
    pcd1_down = pcd1.voxel_down_sample(voxel_size)
    matching01 = get_matching_indices(pcd0_down, pcd1_down, voxel_size)
    matching10 = get_matching_indices(pcd1_down, pcd0_down, voxel_size)
    overlap0 = len(matching01) / len(pcd0_down.points)
    overlap1 = len(matching10) / len(pcd1_down.points)
    return max(overlap0, overlap1)


def get_matching_indices(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, search_voxel_size: float) \
        -> List[Tuple[int, int]]:
    pcd_tree = o3d.geometry.KDTreeFlann(target)

    match_inds = []
    for i, point in enumerate(source.points):
        [_, idx, _] = pcd_tree.search_hybrid_vector_3d(point, search_voxel_size, 1)
        idx = idx[:1]
        for j in idx:
            match_inds.append((i, j))
    return match_inds


def normalize_wrt_first(pcd0: o3d.geometry.PointCloud, pcd1: o3d.geometry.PointCloud):
    translation = -pcd0.get_center()
    return pcd0.translate(translation), pcd1.translate(translation)
