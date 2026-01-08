import numpy as np
PATH_INPUT_MOLECULES = "inputs/one_molecule_structures/"

def auto_create_grid(config):
    """
    In case you want to automatically construct the grid in x and y direction based on crystal cell.
    """
    all_translation_coo = np.array(config["grid"]["translation_subgrids_A"]).reshape(-1)
    if 'None' in all_translation_coo:
        from molgri.molecules.find_unit_cell import get_x_y_grid_inputs
        path_input = f"{PATH_INPUT_MOLECULES}{config['pseudotrajectory']['molecule_1']}.gro"
        num_x_points = config["grid"]["translation_subgrids_A"][0][2]
        num_y_points = config["grid"]["translation_subgrids_A"][1][2]
        z = config["grid"]["translation_subgrids_A"][2]
        x, y = get_x_y_grid_inputs(path_input, num_x_points=num_x_points, num_y_points=num_y_points)
        config["grid"]["translation_subgrids_A"] = [x, y, z]
    return config