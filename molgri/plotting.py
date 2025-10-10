import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
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


def draw_points(points, fig = None, label_by_index: bool = False, custom_labels=None, color="black", **kwargs):
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

    if len(points.shape) == 2 and points.shape[1] == 4:
        # last coordinate of quaternions is shown as opacity
        for point_i, point in enumerate(points):
            opacity = np.abs(point[3])
            if text is not None:
                one_point_text = text[point_i]
            else:
                one_point_text = None
            fig.add_trace(go.Scatter3d(x=(point[0],), y=(point[1],), z=(point[2],), text=one_point_text, mode=mode,
                                       marker=dict(color=color, opacity=opacity)), **kwargs)
    else:
        fig.add_trace(go.Scatter3d(x=points.T[0], y=points.T[1], z=points.T[2], text=text, mode=mode,
                                   marker=dict(color=color)), **kwargs)
    return fig


def draw_line_between(fig, point1, point2, color="black", **kwargs):
    fig.add_trace(
        go.Scatter3d(x=[point1[0], point2[0]],
                     y=[point1[1], point2[1]],
                     z=[point1[2], point2[2]],
                     mode="lines",
                     marker=dict(color=color)), **kwargs)


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


def show_graph(G, node_property: str = "total_index", edge_property: str = "edge_type"):
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
    plt.show()