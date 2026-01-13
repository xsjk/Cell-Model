import numpy as np
import plotly.graph_objects as go
from .voxel import VoxelIsosurface

X, Y, Z = np.ogrid[-1:1:100j, -1:1:100j, -1:1:100j]

fig = go.Figure()
fig.add_traces(
    VoxelIsosurface(
        x_range=(-1, 1),
        y_range=(-1, 1),
        z_range=(-1, 1),
        value=np.sqrt(X**2 + Y**2 + Z**2),
        isomin=0,
        isomax=1,
        surface_count=5,
    )
)

fig.update_layout(
    template="plotly_dark",
    title="Voxel Isosurface Example",
)

fig.write_html("volume_mesh.html", include_plotlyjs="cdn")
