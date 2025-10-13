"""
Everything up to the introduction of molecules: get a full grid, adjacency matrix, surfaces, distances, volumes.
"""
import pickle

ROTATION_ALGORITHM = "hypercube"
N_ROTATION =  25

TRANSLATION_ALGORITHM = "cartesian_nonperiodic"
DEFINE_TRANSLATION_EACH_SUBGRID = ((0, 10, 5), (0, 20, 3), (0, 10, 7))

UNIQUE_GRID_NAME = f"{ROTATION_ALGORITHM}-{N_ROTATION}-{TRANSLATION_ALGORITHM}-test"


rule all:
    input:
        full_network = f"output/networks/{UNIQUE_GRID_NAME}/full_network.pkl"

rule create_full_network:
    output:
        network_file = "{some_path}/full_network.pkl"
    run:
        from molgri.network.rotation_network import create_rotation_network
        from molgri.network.translation_network import create_translation_network
        from molgri.network.full_network import create_full_network
        import networkx as nx

        rotation_network = create_rotation_network(ROTATION_ALGORITHM, N_ROTATION)
        translation_network = create_translation_network(TRANSLATION_ALGORITHM,*DEFINE_TRANSLATION_EACH_SUBGRID)

        full_network = create_full_network(translation_network, rotation_network)

        with open(output.network_file,"wb") as f:
            pickle.dump(full_network, f)


rule display_rotation_properties:
    run:
        import plotly.graph_objects as go
        from molgri.network.rotation_network import create_rotation_network
        from molgri.plotting import draw_points, show_graph, show_array

        ro_network = create_rotation_network(ROTATION_ALGORITHM, N_ROTATION)
        print(ro_network)

        fig = go.Figure()
        draw_points(ro_network.grid,fig,label_by_index=True)
        hull = ro_network.hulls[0]
        draw_points(hull,fig,color="green")
        fig.show()

        show_graph(ro_network, edge_property="distance")
        show_graph(ro_network,edge_property="surface")


        show_array(ro_network.adjacency_type_matrix.toarray(), "Adjacency_type")
        show_array(ro_network.distance_matrix.toarray(), "Distance_matrix")
        show_array(ro_network.surface_matrix.toarray(), "Surface_matrix")

rule display_translation_properties:
    run:
        import plotly.graph_objects as go
        from molgri.network.translation_network import create_translation_network
        from molgri.plotting import draw_points, show_graph, show_array

        to_network = create_translation_network(TRANSLATION_ALGORITHM, *DEFINE_TRANSLATION_EACH_SUBGRID)


        fig = go.Figure()
        draw_points(to_network.grid,fig,label_by_index=True)
        hull = to_network.hulls[0]
        draw_points(hull,fig,color="green")
        fig.show()

        show_graph(to_network, edge_property="distance")
        show_graph(to_network,edge_property="surface")


        show_array(to_network.adjacency_type_matrix.toarray(), "Adjacency_type")
        show_array(to_network.distance_matrix.toarray(), "Distance_matrix")
        show_array(to_network.surface_matrix.toarray(), "Surface_matrix")


