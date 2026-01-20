from MDAnalysis import Universe
from MDAnalysis.topology.guessers import guess_bonds
from plotly import graph_objects as go


def plot_graphene(u: Universe, fig: go.Figure, in_3d:bool=True):
    bonds = guess_bonds(u.select_atoms("all"),u.atoms.positions)
    u.add_TopologyAttr('bonds',bonds)

    positions = u.atoms.positions
    x, y = positions[:, 0], positions[:, 1]
    if in_3d:
        z = positions[:, 2]
    bond_traces = []
    for bond in u.bonds:
        i, j = bond.atoms.indices
        if in_3d:
            bond_traces.append(
                go.Scatter3d(
                    x=[x[i], x[j]],
                    y=[y[i], y[j]],
                    z=[z[i], z[j]],
                    mode='lines+markers',
                    line=dict(color='gray', width=3),
                    hoverinfo='skip'
                ))
        else:
            bond_traces.append(
                go.Scatter(
                    x=[x[i], x[j]],
                    y=[y[i], y[j]],
                    mode='lines+markers',
                    line=dict(color='gray',width=3),
                    hoverinfo='skip'
                ))

    for bt in bond_traces:
        fig.add_trace(bt)
    fig.update_layout(showlegend=False)
