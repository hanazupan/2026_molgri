from workflows.helpers.io import read_object, write_object
import numpy as np

rule read_in_energies:
    input:
        network = "/home/hanaz63/2026_molgri/outputs/networks/graphene_grid_D1/full_network/network.pkl"
    output:
        network_energy = "/home/hanaz63/2026_molgri/outputs/tests/network_energy.pkl"
    run:

        my_network = read_object(input.network)
        random_E = np.random.rand(600)
        print(random_E.shape)

        my_network.add_node_property(random_E, "energy")
        obtained_E = my_network.get_node_property("energy")

        assert np.allclose(random_E, obtained_E)

        write_object(my_network, output.network_energy)

rule get_lowest_E_structures:
    input:
        network_energy = "/home/hanaz63/2026_molgri/outputs/tests/network_energy.pkl"
    output:
        lowest_E =
