"""
These functions can be used directly to quickly access some information
"""

from workflow.helpers.io import read_object

rule print_lowest_energies:
    """
    Use this rule if you want to quickly look at the indices of the lowest energies.
    """
    input:
        energy_csv =f"<pseudosimulation>energy.csv"
    run:
        df = read_object(input.energy_csv)
        df = df.sort_values(by="Binding energy [kJ/mol]",ascending=True)
        print(df.head(50))

rule print_position_assignment:
    input:
        energy_csv = rules.position_assignment_csv.output.assignment_csv
    run:
        df = read_object(input.energy_csv)
        print(df.loc[27])

rule print_position_subgrids:
    input:
        grid_info = rules.save_basic_grid_information.output.info_material
    run:
        grid_info = read_object(input.grid_info)
        print("X gridpoints: ", grid_info["subgrid_points"][0])
        print("Y gridpoints: ", grid_info["subgrid_points"][1])
        print("Z gridpoints: ", grid_info["subgrid_points"][2])


