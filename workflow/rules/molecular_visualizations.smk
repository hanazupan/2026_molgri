"Perform analyses on energy-structure networks, eg identifying and plotting paths."
import matplotlib.pyplot as plt

plt.switch_backend('agg')


rule all_visualization:
    input:
        expand("/home/hanaz63/2026_molgri/outputs/pseudotrajectories/graphene_xylene/graphene_grid_D1/molecular_plots/frame_{i}.tga",
            i=[1, 5, 20])


rule plot_one_frame:
    """
    Plot one specific frame and save it to molecular_plots/
    """
    input:
        structure=f"<outputs_gromacs>structure.gro",
        trajectory=f"<outputs_gromacs>trajectory.xtc",
        structure1=f"<outputs_gromacs>molecule1.gro",
        structure2=f"<outputs_gromacs>molecule2.gro",
        translation_rotation_script = f"<inputs_vmd>script{{view_i}}.log"
    output:
        vmdlog=f"<outputs_vmd>frame_{{frame_index}}_view{{view_i}}",
        frame_plot=f"<outputs_frame_plots>frame_{{frame_index}}_view{{view_i}}.tga"
    run:
        from molgri.create_vmdlog import VMDCreator
        from workflow.helpers.io import get_num_atoms

        n1 = get_num_atoms(input.structure1)
        n2 = get_num_atoms(input.structure2)

        my_vmd = VMDCreator(f"index < {n1}", f"index >= {n1}")
        my_vmd.load_translation_rotation_script(input.translation_rotation_script)

        index_to_plot = [int(wildcards.frame_index) + 1]

        my_vmd.plot_these_structures(index_to_plot,[output.frame_plot])
        my_vmd.write_text_to_file(output.vmdlog)

        shell("vmd  -dispdev text {input.structure} {input.trajectory} < {output.vmdlog}")


rule plot_overlay_frames:
    """
    Plot multiple frames at the same time (eg. 10 lowest E structures, but this is a general rule).
    """
    input:
        structure=f"<outputs_gromacs>structure.gro",
        trajectory=f"<outputs_gromacs>trajectory.xtc",
        structure1=f"<outputs_gromacs>molecule1.gro",
        structure2=f"<outputs_gromacs>molecule2.gro",
        translation_rotation_script = f"<inputs_vmd>script{{view_i}}.log",
        indices= f"<outputs_indices>{{file_name}}.txt"
    output:
        vmdlog=f"<outputs_vmd>{{file_name}}_view{{view_i}}",
        frame_plot=f"<outputs_plots_lowest>{{file_name}}_view{{view_i}}.tga"
    run:
        from molgri.create_vmdlog import VMDCreator
        from workflow.helpers.io import get_num_atoms, read_object

        indices = read_object(input.indices)

        n1 = get_num_atoms(input.structure1)
        n2 = get_num_atoms(input.structure2)

        my_vmd = VMDCreator(f"index < {n1}", f"index >= {n1}")
        my_vmd.load_translation_rotation_script(input.translation_rotation_script)

        index_to_plot = [int(i) + 1 for i in indices]

        my_vmd.plot_multiple_overlappig_frames(index_to_plot,output.frame_plot)
        my_vmd.write_text_to_file(output.vmdlog)

        shell("vmd  -dispdev text {input.structure} {input.trajectory} < {output.vmdlog}")
