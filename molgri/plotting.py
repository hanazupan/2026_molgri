import numpy as np
from scipy.spatial import geometric_slerp
import plotly.graph_objects as go

from molgri.utils import normalise_vectors


def draw_curve(fig, start_point, end_point, color="black"):
    norm = np.linalg.norm(start_point)
    interpolate_points = np.linspace(0, 1, 2000)
    curve = geometric_slerp(normalise_vectors(start_point), normalise_vectors(end_point), interpolate_points)
    fig.add_trace(go.Scatter3d(x=norm * curve[..., 0], y=norm * curve[..., 1], z=norm * curve[..., 2],
                               mode="lines", line=dict(color=color)))