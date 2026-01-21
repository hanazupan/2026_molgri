"""
Here are the rules that are specific to rerun and pseudotrajectory so they are not confused with e.g. how a trajectory
is generated in a real trajectory.
"""

from workflow.helpers.io import read_object, write_object

rule create_pseudotrajectory:
    """
    Here we are creating a pseudotrajectory from two molecules and a network.
    """
    input:
        molecule_1 = f"<outputs_gromacs>molecule1.<ext_str>",
        molecule_2 = f"<outputs_gromacs>molecule2.<ext_str>",
        network = f"<outputs_network>network.pkl"
    output:
        trajectory = f"<outputs_gromacs>trajectory.<ext_trj>"
    run:
        from molgri.molecules.bimolecular import get_bimolecular_pseudotrajectory, move_to_center
        m1 = read_object(input.molecule_1)
        m2 = read_object(input.molecule_2)


        network = read_object(input.network)
        weights = m2.atoms.masses
        coordinates = network.create_pseudotrajectory_coordinates_from(m2.atoms.positions, weights)

        pt = get_bimolecular_pseudotrajectory(m1, m2, coordinates)
        write_object(pt, output.trajectory)


rule gromacs_rerun:
    """
    This rule gets structure, trajectory, topology and gromacs run file as input, as output we are only interested in
    energies.
    """
    input:
        structure = f"<outputs_gromacs>structure.<ext_str>",
        trajectory = f"<outputs_gromacs>trajectory.<ext_trj>",
        runfile = f"<outputs_gromacs>production.mdp",
        topology = f"<outputs_gromacs>topol.top",
        index=f"<outputs_gromacs>index.ndx",
        select_energy = f"<outputs_gromacs>select_energy",
        force_field_stuff = f"<outputs_gromacs>force_field_stuff/"
    shadow: "minimal"
    log:
        log = f"<outputs_gromacs>logging_gromacs.log"
    benchmark:
        repeat(f"<outputs_gromacs>gromacs_benchmark.txt", 1)
    output:
        energy = f"<outputs_gromacs>energy.xvg",
    shell:
        """
        initial_dir=$(pwd)
        cd $(dirname {input.runfile})
        export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
        gmx22 grompp -f $(basename {input.runfile}) -c $(basename {input.structure}) -r $(basename {input.structure}) -p $(basename {input.topology}) -o result.tpr  -n $(basename {input.index})
        gmx22 mdrun -s result.tpr -rerun $(basename {input.trajectory}) -g $(basename {log.log})
        gmx22 energy -f ener.edr -o $(basename {output.energy}) < $(basename {input.select_energy})
        cd "$initial_dir" || exit
        """


rule read_in_energies:
    """
    Here we can assign the energy to the nodes of the network.
    """
    input:
        network = f"<outputs_network>network.pkl",
        energy = f"<outputs>energy.csv",
    output:
        network_energy = f"<outputs>network_energy.pkl"
    run:

        my_network = read_object(input.network)
        my_energy = read_object(input.energy)
        my_energy_array = my_energy["Binding energy [kJ/mol]"].to_numpy()
        my_network.add_node_properties(my_energy_array,"binding_energy")
        write_object(my_network, output.network_energy)



rule plot_overlay_all_translations:
    """
    Show a VMD plot that overlaps all possible rotations of molecule2 at the first grid position.
    """
    input:
        frame_plot = f"<overlayed_vmd_frames>all_positions_first_rotation_view3_full.tga",
        frame_plot_COM = f"<overlayed_vmd_frames>all_positions_first_rotation_view3_COM.tga"


rule plot_overlay_all_rotations:
    """
    Show a VMD plot that overlaps all possible positions of molecule2 at the first orientation.
    """
    input:
        frame_plot= f"<overlayed_vmd_frames>all_rotations_first_position_view3_full.tga",
        frame_plot_COM= f"<overlayed_vmd_frames>all_rotations_first_position_view3_COM.tga"


rule stack_all_rotation_options:
    """
    Collect images of each rotation and plot them next to each other.
    """
    input:
        joint_image_both = "<stacked_vmd_frames>all_rotations_first_position_both.png",
        joint_image_m2 = "<stacked_vmd_frames>all_rotations_first_position_m2.png"

rule stack_all_translation_options:
    """
    Collect images of each translation and plot them next to each other. Warning, this can be a looot of subplots.
    """
    input:
        joint_image_both = "<stacked_vmd_frames>all_positions_first_rotation_both.png",
        joint_image_m2 = "<stacked_vmd_frames>all_positions_first_rotation_m2.png"


