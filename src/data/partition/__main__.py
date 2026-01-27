import numpy as np
import plotly.graph_objects as go
from codetiming import Timer
from plotly.subplots import make_subplots

from src.visualization.objects import VoxelIsosurface

from . import partition_edt, partition_laplace

# Load masks
cell_mask: np.ndarray = np.load("cell_mask.npy") > 0
nucleus_mask: np.ndarray = np.load("nucleus_mask.npy") > 0
## Pad input masks so that the cell is viewed as unclipped at the voxel grid boundary
cell_mask = np.pad(cell_mask, pad_width=1, mode="constant", constant_values=0)
nucleus_mask = np.pad(nucleus_mask, pad_width=1, mode="constant", constant_values=0)
cytoplasm_mask = cell_mask & (~nucleus_mask)
combined_masks = cell_mask.astype(float) + nucleus_mask.astype(float)
assert np.all(cell_mask[nucleus_mask]), "Nucleus must be inside cell"
print(f"Mask shape: {cell_mask.shape}")


# Compute partitions
print("Partitioning fields...")
with Timer(text="EDT:               {:.4f}s"):
    s_edt = partition_edt(cell_mask, nucleus_mask)
with Timer(text="Laplace:           {:.4f}s"):
    s_lap = partition_laplace(cell_mask, nucleus_mask)
with Timer(text="EDT Equalized:     {:.4f}s"):
    s_edt_eq = partition_edt(cell_mask, nucleus_mask, equalization=True)
with Timer(text="Laplace Equalized: {:.4f}s"):
    s_lap_eq = partition_laplace(cell_mask, nucleus_mask, equalization=True)


# Visualization
def add_iso(r, c, vol, surface_count=10, opacity=0.1):
    fig.add_traces(
        VoxelIsosurface(
            value=vol,
            mask=cytoplasm_mask,
            isomin=0,
            isomax=1,
            surface_count=surface_count,
            opacity=opacity,
            colorscale="Viridis",
            showscale=False,
        ),
        rows=r,
        cols=c,
    )


## Isosurface comparison
Z, Y, X = cell_mask.shape
max_dim = max(X, Y, Z)
scene_cfg = dict(
    aspectratio=dict(x=X / max_dim, y=Y / max_dim, z=Z / max_dim * 3),
    xaxis=dict(title="X", nticks=5, range=[0, X]),
    yaxis=dict(title="Y", nticks=5, range=[0, Y]),
    zaxis=dict(title="Z", nticks=5, range=[0, Z]),
)
fig = make_subplots(
    rows=1,
    cols=5,
    specs=[[{"type": "scene"}] * 5],
    subplot_titles=("Cell & Nucleus Masks", "EDT", "EDT Equalized", "Laplace", "Laplace Equalized"),
    horizontal_spacing=0.01,
)
add_iso(1, 1, combined_masks, surface_count=3, opacity=0.3)
add_iso(1, 2, s_edt)
add_iso(1, 2, combined_masks, surface_count=3, opacity=0.05)
add_iso(1, 3, s_edt_eq)
add_iso(1, 3, combined_masks, surface_count=3, opacity=0.05)
add_iso(1, 4, s_lap)
add_iso(1, 4, combined_masks, surface_count=3, opacity=0.05)
add_iso(1, 5, s_lap_eq)
add_iso(1, 5, combined_masks, surface_count=3, opacity=0.05)
fig.update_layout(
    template="plotly_dark",
    title_text="Partition Mask Algorithms Comparison",
)
fig.update_scenes(scene_cfg)
fig.write_html("partition_comparison.html", include_plotlyjs="cdn")


## Slice comparison
z_slice = s_edt.shape[0] // 2
fig_slice = make_subplots(
    rows=2,
    cols=2,
    specs=[[{"type": "scene"}] * 2] * 2,
    subplot_titles=("EDT", "Laplace", "EDT Equalized", "Laplace Equalized"),
    horizontal_spacing=0.01,
    vertical_spacing=0.05,
)
fig_slice.add_trace(go.Surface(z=s_edt[z_slice], showscale=False, colorscale="Viridis"), row=1, col=1)
fig_slice.add_trace(go.Surface(z=s_lap[z_slice], showscale=False, colorscale="Viridis"), row=1, col=2)
fig_slice.add_trace(go.Surface(z=s_edt_eq[z_slice], showscale=False, colorscale="Viridis"), row=2, col=1)
fig_slice.add_trace(go.Surface(z=s_lap_eq[z_slice], showscale=False, colorscale="Viridis"), row=2, col=2)
fig_slice.update_layout(
    template="plotly_dark",
    title_text=f"Partition Field Slice Comparison (z={z_slice})",
)
fig_slice.write_html("partition_slice_comparison.html", include_plotlyjs="cdn")


## Histogram comparison
fig_hist = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=("Original Scalar Fields", "Equalized Scalar Fields"),
    vertical_spacing=0.05,
)
fig_hist.add_trace(go.Histogram(x=s_edt[cytoplasm_mask], name="EDT", opacity=0.75), row=1, col=1)
fig_hist.add_trace(go.Histogram(x=s_lap[cytoplasm_mask], name="Laplace", opacity=0.75), row=1, col=1)
fig_hist.add_trace(go.Histogram(x=s_edt_eq[cytoplasm_mask], name="EDT Eq", opacity=0.75), row=2, col=1)
fig_hist.add_trace(go.Histogram(x=s_lap_eq[cytoplasm_mask], name="Laplace Eq", opacity=0.75), row=2, col=1)
fig_hist.update_layout(title_text="Scalar Field Histogram Comparison", template="plotly_dark", barmode="overlay")
fig_hist.write_html("partition_hist.html", include_plotlyjs="cdn")
