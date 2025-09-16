import numpy as np
from scipy.spatial import geometric_slerp
import plotly.graph_objects as go
import plotly.express as px

from molgri.utils import normalise_vectors, sort_points_on_sphere_ccw


def draw_curve(fig, start_point, end_point, color="black"):
    norm = np.linalg.norm(start_point)
    interpolate_points = np.linspace(0, 1, 2000)
    curve = geometric_slerp(normalise_vectors(start_point), normalise_vectors(end_point), interpolate_points)
    fig.add_trace(go.Scatter3d(x=norm * curve[..., 0], y=norm * curve[..., 1], z=norm * curve[..., 2],
                               mode="lines", line=dict(color=color)))


def draw_spherical_polygon(fig, points, color="black"):
    if len(points) < 3:
        # no poligon, quietly exit
        return
    sorted_points = sort_points_on_sphere_ccw(points)
    # duplicate first point as last point to draw a closed curve
    sorted_points = np.vstack([sorted_points, sorted_points[0]])
    # draw all the arches
    for point1, point2 in zip(sorted_points, sorted_points[1:]):
        draw_curve(fig, point1, point2, color=color)


def draw_points(fig, points, label_by_index: bool = False, color="black"):
    if label_by_index:
        text = list(range(len(points)))
        fig.add_trace(go.Scatter3d(x=points.T[0], y=points.T[1], z=points.T[2], text=text, mode="text+markers",
                         marker=dict(color=color)))
    else:
        fig.add_trace(go.Scatter3d(x=points.T[0], y=points.T[1], z=points.T[2], mode="markers",
                                   marker=dict(color=color)))


def draw_line_between(fig, point1, point2, color="black"):
    fig.add_trace(
        go.Scatter3d(x=[point1[0], point2[0]],
                     y=[point1[1], point2[1]],
                     z=[point1[2], point2[2]],
                     mode="lines",
                     marker=dict(color=color)))


def show_array(my_array, title: str = ""):
    fig = px.imshow(my_array)
    fig.update_layout(
        font=dict(size=30),
        title={
            'text': title,
            'y': 1,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    fig.show()
