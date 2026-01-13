from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType
from skimage.measure import marching_cubes


@dataclass(kw_only=True)
class VoxelIsosurface(list[BaseTraceType]):
    """
    A lightweight replacement for go.Volume that exports only triangle meshes instead of full voxel data.
    It converts a dense 3D volume into Mesh3d traces.
    """

    value: np.ndarray
    mask: np.ndarray | None = None
    isomin: float
    isomax: float
    surface_count: int
    opacity: float = 0.25
    colorscale: str = "Plotly3"
    showscale: bool = True
    colorbar_title: str | None = None
    colorbar: dict | None = None
    x_range: tuple[float, float] | None = None
    y_range: tuple[float, float] | None = None
    z_range: tuple[float, float] | None = None
    max_triangles_per_surface: int | None = None
    name: str | None = None

    def __post_init__(self):
        super().__init__()

        if self.value.ndim != 3:
            raise ValueError(f"value must be 3D (Z,Y,X). Got shape={self.value.shape}")

        # Calculate effective spacing and origin based on ranges
        nz, ny, nx = self.value.shape
        dz, dy, dx = 1.0, 1.0, 1.0
        oz, oy, ox = 0.0, 0.0, 0.0

        if self.z_range is not None:
            z_min, z_max = self.z_range
            dz = (z_max - z_min) / max(1, nz - 1)
            oz = z_min
        if self.y_range is not None:
            y_min, y_max = self.y_range
            dy = (y_max - y_min) / max(1, ny - 1)
            oy = y_min
        if self.x_range is not None:
            x_min, x_max = self.x_range
            dx = (x_max - x_min) / max(1, nx - 1)
            ox = x_min

        spacing = (dz, dy, dx)
        origin = np.array([oz, oy, ox], dtype=np.float32)

        if self.surface_count < 1:
            raise ValueError("surface_count must be >= 1.")

        if self.surface_count == 1:
            levels = np.array([(self.isomin + self.isomax) * 0.5], dtype=np.float32)
        else:
            levels = np.linspace(self.isomin, self.isomax, self.surface_count, dtype=np.float32)

        cmin, cmax = levels.min(), levels.max()

        for idx, iso in enumerate(levels):
            try:
                verts, faces, _, _ = marching_cubes(self.value, level=iso, spacing=spacing, mask=self.mask)
                verts += origin
            except Exception:
                continue

            intensity = np.full((verts.shape[0],), iso, dtype=np.float32)
            opacity = self.opacity * (0.6 + 0.4 * idx / (len(levels) - 1))
            name = f"{self.name}: iso={float(iso):.3f}" if self.name else None

            self.append(
                go.Mesh3d(
                    x=verts[:, 2],
                    y=verts[:, 1],
                    z=verts[:, 0],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    intensity=intensity,
                    colorscale=self.colorscale,
                    cmin=cmin,
                    cmax=cmax,
                    opacity=opacity,
                    showscale=False,
                    name=name,
                )
            )

        # Build a single shared colorbar using a dummy trace
        if self.showscale:
            self.append(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode="markers",
                    marker=dict(
                        size=0.1,
                        color=[cmin, cmax],
                        colorscale=self.colorscale,
                        cmin=cmin,
                        cmax=cmax,
                        showscale=True,
                        colorbar=self.colorbar,
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
