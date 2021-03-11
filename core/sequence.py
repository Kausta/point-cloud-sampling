import os
from typing import Tuple

import numpy as np
import open3d as o3d


class Sequence:
    def __init__(self, data_root: str, name: str, filename: str, transform: Tuple[float, float, float] = None):
        self.name = name
        self.data_root = data_root
        self.filename = filename
        self.path = os.path.join(data_root, filename)
        self.transform = transform
        self._pcd = None

    def _read_pcd(self):
        self._pcd = o3d.io.read_point_cloud(self.path)
        if self.transform is not None:
            translation_vec = np.array(self.transform, dtype=np.float64)
            self._pcd = self._pcd.translate(translation_vec)
        print(f"Read pcd {self}:\n\t{self._pcd}")

    def _check_pcd(self):
        if self._pcd is None:
            self._read_pcd()

    def get_pcd(self) -> o3d.geometry.PointCloud:
        self._check_pcd()
        return self._pcd

    def get_pcd_array(self) -> np.ndarray:
        self._check_pcd()
        return np.asarray(self._pcd.points)

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        self._check_pcd()
        bbox = self._pcd.get_axis_aligned_bounding_box()
        return bbox.get_min_bound(), bbox.get_max_bound()

    def __repr__(self) -> str:
        return f"Sequence \"{self.name}\" at \"{self.path}\""
