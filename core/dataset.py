import os
from typing import Tuple, List

import numpy as np
import open3d as o3d
from tqdm import tqdm

from .util import get_rectangle_pcd, compute_overlap_ratio
from .sequence import Sequence


class Dataset:
    def __init__(self, name: str,
                 first_seq: Sequence, second_seq: Sequence,
                 rect_size: Tuple[float, float], interval: Tuple[float, float], min_percent: float,
                 out_dir: str, normalize_output: bool = False):
        self.name = name
        self.first_seq = first_seq
        self.second_seq = second_seq
        self.rect_size = np.array(rect_size)
        self.interval = np.array(interval)
        self.min_percent = min_percent
        self.out_dir = out_dir
        self.normalize_output = normalize_output

    def _calculate_grid(self) -> np.ndarray:
        min_b1, max_b1 = self.first_seq.get_bounds()
        min_b2, max_b2 = self.second_seq.get_bounds()
        min_bound, max_bound = np.amin([min_b1, min_b2], axis=0), np.amax([max_b1, max_b2], axis=0)
        print("Minimum bound:", min_bound)
        print("Maximum bound:", max_bound)
        x_starts = np.arange(min_bound[0], max_bound[0] - self.rect_size[0] / 2, self.interval[0])
        y_starts = np.arange(min_bound[1], max_bound[1] - self.rect_size[1] / 2, self.interval[1])
        grid = np.transpose([np.tile(x_starts, len(y_starts)), np.repeat(y_starts, len(x_starts))])
        return grid

    def _find_all_pairs(self, minxy_points: np.ndarray) \
            -> List[Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]]:
        first_pcd_points = self.first_seq.get_pcd_array()
        second_pcd_points = self.second_seq.get_pcd_array()
        voxel_pairs = []
        print(f"Reading pairs, {len(minxy_points)} candidates")
        for minxy in tqdm(minxy_points):
            voxel1 = get_rectangle_pcd(first_pcd_points, minxy, self.rect_size)
            if voxel1 is None:
                continue
            voxel2 = get_rectangle_pcd(second_pcd_points, minxy, self.rect_size)
            if voxel2 is None:
                continue
            voxel_pairs.append((voxel1, voxel2))
        print(f"Read {len(voxel_pairs)} pairs")
        return voxel_pairs

    def _get_valid_pairs(self, voxel_pairs: List[Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]],
                         matching_search_voxel_size: float = 0.2) \
            -> List[Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, float]]:
        print("Filtering valid pairs")
        pairs = []
        min_overlap_ratio, running_overlap_ratio, max_overlap_ratio = 1.1, 0, -0.1
        overall_min, overall_max = 1.1, -0.1

        for pcd1, pcd2 in tqdm(voxel_pairs):
            overlap_ratio = compute_overlap_ratio(pcd1, pcd2, matching_search_voxel_size)
            overall_min = min(overall_min, overlap_ratio)
            overall_max = max(overall_max, overlap_ratio)
            if self.min_percent <= overlap_ratio:
                pairs.append((pcd1, pcd2, overlap_ratio))
                min_overlap_ratio = min(min_overlap_ratio, overlap_ratio)
                running_overlap_ratio += overlap_ratio
                max_overlap_ratio = max(max_overlap_ratio, overlap_ratio)

        print(f"Filtered {len(pairs)} valid pairs")
        print(f"Overall Min Overlap: {overall_min:.4f}")
        print(f"Overall Max Overlap: {overall_max:.4f}")

        if len(pairs) != 0:
            avg_overlap_ratio = running_overlap_ratio / len(pairs)
            print(f"Min Overlap Ratio: {min_overlap_ratio:.4f}")
            print(f"Max Overlap Ratio: {max_overlap_ratio:.4f}")
            print(f"Avg Overlap Ratio: {avg_overlap_ratio:.4f}")

        return pairs

    def _save_voxel(self, filename: str, pcd: o3d.geometry.PointCloud):
        filename = os.path.join(self.out_dir, filename)
        o3d.io.write_point_cloud(f"{filename}.ply", pcd)
        np.savez(f"{filename}.npz", pcd=np.asarray(pcd.points))

    def _save_processed_pairs(self, pairs: List[Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, float]]):
        print("Writing pair files")
        filenames = []
        for i, (pcd1, pcd2, ratio) in enumerate(tqdm(pairs)):
            filename1 = f"{self.name}@seq-{self.first_seq.name}_{str(i).zfill(3)}"
            filename2 = f"{self.name}@seq-{self.second_seq.name}_{str(i).zfill(3)}"
            filenames.append((f"{filename1}.npz", f"{filename2}.npz", ratio))

            self._save_voxel(filename1, pcd1)
            self._save_voxel(filename2, pcd2)

        dset_file = os.path.join(self.out_dir, f"dataset-{self.name}.txt")
        with open(dset_file, 'w') as f:
            for f1, f2, rat in filenames:
                f.write(f"{f1} {f2} {rat}\n")

    def process(self):
        minxy_points = self._calculate_grid()
        voxel_pairs = self._find_all_pairs(minxy_points)
        valid_pairs = self._get_valid_pairs(voxel_pairs)
        self._save_processed_pairs(valid_pairs)

    def __repr__(self) -> str:
        from pprint import pformat
        return pformat(vars(self), indent=4, width=1)
