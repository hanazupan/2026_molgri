"""
Everything up to the introduction of molecules: get a full grid, adjacency matrix, surfaces, distances, volumes.
"""

from workflows.helpers.PATHS import PATH_OUTPUT_NETWORKS
from workflows.helpers.io import write_object, read_object

ROTATION_ALGORITHM = config["rotation_algorithm"]
N_ROTATION =  config["N_rotations"]
TRANSLATION_ALGORITHM = config["translation_algorithm"]
DEFINE_TRANSLATION_EACH_SUBGRID = config["translation_subgrids_A"]
NETWORK_ID = config["unique_network_name"]


import matplotlib
matplotlib.use('Agg')

rule all:
    input:
        expand(f"{PATH_OUTPUT_NETWORKS}{NETWORK_ID}/{{network_type}}_network/{{to_create}}",
            network_type = ["full", "rotation", "translation"],
               to_create =["network.pkl", "adjacency.npz", "distances.npz", "surfaces.npz", "edge_types.npz", "grid.npy", "volumes.npy"]),

        # optional visualizations
        expand(f"{PATH_OUTPUT_NETWORKS}{NETWORK_ID}/{{network_type}}_network/{{to_create}}.{{ending}}",
            ending = ["png"],
            network_type = ["full", "rotation", "translation"],
               to_create =["adjacency", "distances", "surfaces", "edge_types", "grid", "volumes", "network"])

rule create_rotation_network:
    benchmark:
        "{some_path}/rotation_network/network_creation.txt"
    output:
        network_file = "{some_path}/rotation_network/network.pkl",
    run:
        from molgri.network.rotation_network import create_rotation_network

        rotation_network = create_rotation_network(ROTATION_ALGORITHM, N_ROTATION)
        write_object(rotation_network, output.network_file)

rule create_translation_network:
    benchmark:
        "{some_path}/translation_network/network_creation.txt"
    output:
        network_file = "{some_path}/translation_network/network.pkl",
    run:
        from molgri.network.translation_network import create_translation_network

        translation_network = create_translation_network(TRANSLATION_ALGORITHM, *DEFINE_TRANSLATION_EACH_SUBGRID)
        write_object(translation_network, output.network_file)

rule create_full_network:
    input:
        rotation_network_file = "{some_path}/rotation_network/network.pkl",
        translation_network_file = "{some_path}/translation_network/network.pkl",
    benchmark:
        "{some_path}/full_network/network_creation.txt"
    output:
        network_file = "{some_path}/full_network/network.pkl",
    run:
        from molgri.network.full_network import create_full_network

        rotation_network = read_object(input.rotation_network_file)
        translation_network = read_object(input.translation_network_file)
        full_network = create_full_network(translation_network, rotation_network)
        write_object(full_network, output.network_file)

rule save_network_properties:
    input:
        network_file = "{some_path}/network.pkl"
    benchmark:
        "{some_path}/saving_properties.txt"
    output:
        grid = "{some_path}/grid.npy",
        adjacency = "{some_path}/adjacency.npz",
        numerical_edge_type = "{some_path}/edge_types.npz",
        distances = "{some_path}/distances.npz",
        surfaces = "{some_path}/surfaces.npz",
        volumes = "{some_path}/volumes.npy",
    run:
        full_network = read_object(input.network_file)

        write_object(full_network.grid, output.grid)
        write_object(full_network.volumes, output.volumes)

        write_object(full_network.adjacency_matrix, output.adjacency)
        write_object(full_network.adjacency_type_matrix,output.numerical_edge_type)
        write_object(full_network.distance_matrix,output.distances)
        write_object(full_network.surface_matrix,output.surfaces)

rule display_network:
    input:
        network_file = "{some_path}/network.pkl"
    output:
        plot = "{some_path}/network.png"
    run:
        from molgri.plotting import show_graph

        my_network = read_object(input.network_file)
        show_graph(my_network,edge_property="distance", show=False, save_as=output.plot)



rule display_network_edge_matrices:
    input:
        adjacency = "{some_path}/adjacency.npz",
        numerical_edge_type = "{some_path}/edge_types.npz",
        distances = "{some_path}/distances.npz",
        surfaces = "{some_path}/surfaces.npz",
    output:
        adjacency = "{some_path}/adjacency.png",
        numerical_edge_type = "{some_path}/edge_types.png",
        distances = "{some_path}/distances.png",
        surfaces = "{some_path}/surfaces.png",
        interactive_adjacency= "{some_path}/adjacency.html",
        interactive_numerical_edge_type= "{some_path}/edge_types.html",
        interactive_distances= "{some_path}/distances.html",
        interactive_surfaces= "{some_path}/surfaces.html",
    run:
        from molgri.plotting import show_array

        show_array(read_object(input.adjacency).toarray(), "Adjacency_type",
            save_as=output.adjacency, save_interactive_as=output.interactive_adjacency, show=False)
        show_array(read_object(input.numerical_edge_type).toarray(),"Edge types",
            save_as=output.numerical_edge_type, save_interactive_as=output.interactive_numerical_edge_type, show=False)
        show_array(read_object(input.distances).toarray(), "Distance_matrix",
            save_as=output.distances, save_interactive_as=output.interactive_distances, show=False)
        show_array(read_object(input.surfaces).toarray(), "Surface_matrix",
            save_as=output.surfaces, save_interactive_as=output.interactive_surfaces, show=False)

rule display_network_node_attributes:
    input:
        grid = "{some_path}/grid.npy",
        volumes= "{some_path}/volumes.npy",
    output:
        grid = "{some_path}/grid.png",
        volumes = "{some_path}/volumes.png",
        interactive_grid= "{some_path}/grid.html",
        interactive_volumes= "{some_path}/volumes.html",
    run:
        from molgri.plotting import draw_points
        import numpy as np
        grid = read_object(input.grid)
        draw_points(grid, save_as=output.grid, save_interactive_as=output.interactive_grid, show=False)
        volumes = read_object(input.volumes)
        draw_points(grid, custom_labels=np.round(volumes,2), save_as=output.volumes, marker_size=volumes,
            save_interactive_as=output.interactive_volumes,show=False)


