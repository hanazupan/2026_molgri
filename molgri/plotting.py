from functools import wraps

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import geometric_slerp
import plotly.graph_objects as go
import plotly.express as px

from molgri.utils import normalise_vectors, sort_points_on_sphere_ccw


def normalize_marker_sizes(values, min_size=5, max_size=20):
    """Normalize an array of numbers to marker sizes between min_size and max_size"""
    vmin = np.min(values)
    vmax = np.max(values)
    if np.isclose(vmax, vmin):  # all values are the same
        return np.full_like(values, (min_size + max_size) / 2)
    norm_sizes = (values - vmin) / (vmax - vmin)  # scale to 0-1
    norm_sizes = norm_sizes * (max_size - min_size) + min_size
    return norm_sizes

def save_plotly(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        save_as = kwargs.pop("save_as", None)
        save_interactive_as = kwargs.pop("save_interactive_as", None)
        show = kwargs.pop("show", True)
        fig = func(*args, **kwargs)
        if fig is None:
            return None
        if show:
            fig.show()
        if save_as is not None:
            fig.write_image(save_as)
        if save_interactive_as is not None:
            fig.write_html(save_interactive_as)
        return fig
    return wrapper




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

@save_plotly
def draw_points(points, fig = None, label_by_index: bool = False, custom_labels=None, marker_size=None, color="black",
                **kwargs):
    if fig is None:
        fig = go.Figure()
    if label_by_index or custom_labels is not None:
        if custom_labels is None:
            custom_labels = list(range(len(points)))
        mode = "text+markers"
        text = custom_labels
    else:
        mode = "markers"
        text = None
    if marker_size is None:
        normalized_marker_size = 5
    else:
        normalized_marker_size = normalize_marker_sizes(marker_size)
    if len(points.shape) == 2 and points.shape[1] == 4:
        # last coordinate of quaternions is shown as opacity
        for point_i, point in enumerate(points):
            opacity = np.abs(point[3])
            if text is not None:
                one_point_text = text[point_i]
            else:
                one_point_text = None
            if isinstance(normalized_marker_size, np.ndarray):
                single_marker_size = normalized_marker_size[point_i]
            else:
                single_marker_size = normalized_marker_size
            fig.add_trace(go.Scatter3d(x=(point[0],), y=(point[1],), z=(point[2],), text=one_point_text, mode=mode,
                                       marker=dict(color=color, opacity=opacity, size=single_marker_size)), **kwargs)
    else:
        fig.add_trace(go.Scatter3d(x=points.T[0], y=points.T[1], z=points.T[2], text=text, mode=mode,
                                   marker=dict(color=color, size=normalized_marker_size)), **kwargs)
    return fig


def draw_line_between(fig, point1, point2, color="black", **kwargs):
    fig.add_trace(
        go.Scatter3d(x=[point1[0], point2[0]],
                     y=[point1[1], point2[1]],
                     z=[point1[2], point2[2]],
                     mode="lines",
                     marker=dict(color=color)), **kwargs)

@save_plotly
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
    return fig


def show_graph(G, node_property: str = "total_index", edge_property: str = "edge_type",
               save_as = None, show=True):
    labels = {node: node_i for node_i, node in enumerate(sorted(G.nodes))}

    if edge_property == "edge_type":
        edge_labels = {(u,v): edge_data[edge_property] for u, v, edge_data in G.edges(data=True)}
    else:
        edge_labels = {(u, v): np.round(edge_data[edge_property], 2) for u, v, edge_data in G.edges(data=True)}


    type_to_color = {"radial": "red", "spherical": "blue", "rotational": "yellow",
                     "x": "black", "y": "violet", "z": "orange"}
    edge_colors = [type_to_color[edge_data["edge_type"]] for u, v, edge_data in G.edges(data=True)]

    pos = nx.kamada_kawai_layout(G, weight="numerical_edge_type")
    nx.draw(G, pos, labels=labels)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    if save_as is not None:
        plt.savefig(save_as)
    if show:
        plt.show()
