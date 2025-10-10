"""
Everything up to the introduction of molecules: get a full grid, adjacency matrix, surfaces, distances, volumes.
"""
import numpy as np


ROTATION_ALGORITHM = "hypercube"
N_ROTATION =  25

TRANSLATION_ALGORITHM = "cartesian_periodic"
DEFINE_TRANSLATION_EACH_SUBGRID = ((0, 1, 5), (-1, 3, 3), (1, 2, 7))


rule display_translation_properties:
    run:
        import plotly.graph_objects as go
        from molgri.translation_network import create_translation_network
        from molgri.plotting import draw_points, show_array, show_graph

        to_network = create_translation_network(TRANSLATION_ALGORITHM, *DEFINE_TRANSLATION_EACH_SUBGRID)


        fig = go.Figure()
        draw_points(to_network.grid,fig,label_by_index=True)
        hull = to_network.hulls[0]
        draw_points(hull,fig,color="green")
        fig.show()

        show_graph(to_network, edge_property="distance")
        show_graph(to_network,edge_property="surface")



rule display_rotation_properties:
    run:
        import plotly.graph_objects as go

        from molgri.rotgrid import create_rotation_object
        from molgri.plotting import draw_points, show_array, show_graph
        rotation_object = create_rotation_object(N_ROTATION,ROTATION_ALGORITHM)

        # draw the points of the grid and optionally their hulls
        fig = go.Figure()
        draw_points(rotation_object.grid,fig,label_by_index=True)
        hull = rotation_object.hulls[0]
        draw_points(hull,fig,color="green")
        fig.show()

        # show adjacency array
        show_array(rotation_object.adjacency.toarray())

        # plot the graph
        graph_of_ro = rotation_object.get_rotation_network()
        show_graph(graph_of_ro, edge_property="distance")
        show_graph(graph_of_ro,edge_property="surface")


rule create_full_network:
    run:
        from molgri.rotgrid import create_rotation_object
        rotation_network = create_rotation_object(N_ROTATION, ROTATION_ALGORITHM).get_rotation_network()
        translation_network = create_translation_network(TRANSLATION_ALGORITHM,*DEFINE_TRANSLATION_EACH_SUBGRID)