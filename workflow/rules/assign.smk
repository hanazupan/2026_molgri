from workflow.helpers.PATHS import NAME_SIMULATION_FOLDER

rule align_trajectory_to_molecule1:
    """
    The goal here is to fit the molecule 1 during the trajectory as much as possible to the reference structure.
    """
    input:
        reference_full_structure = f"{{some_path}}{NAME_SIMULATION_FOLDER}structure.{STRUCTURE_ENDING}",
        molecule_1 = f"{{some_path}}{NAME_SIMULATION_FOLDER}molecule1.{STRUCTURE_ENDING}",
        trajectory = f"{{some_path}}{NAME_SIMULATION_FOLDER}trajectory.{TRAJECTORY_ENDING}"