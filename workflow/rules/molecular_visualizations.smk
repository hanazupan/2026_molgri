"Perform analyses on energy-structure networks, eg identifying and plotting paths."
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('agg')

from workflow.helpers.PATHS import NAME_PT_FOLDER, NAME_VMD_OUTPUT, NAME_FRAME_PLOTS, PATH_VMD_SCRIPTS, NAME_PLOTS
MOLECULE_NAMES = f"{config['pseudotrajectory']['molecule_1']}_{config['pseudotrajectory']['molecule_2']}/"

rule all_visualization:
    input:
        expand("/home/hanaz63/2026_molgri/outputs/pseudotrajectories/graphene_xylene/graphene_grid_D1/molecular_plots/frame_{i}.tga",
            i=[1, 5, 20])


rule plot_one_frame:
    """
    Plot one specific frame and save it to molecular_plots/
    """
    input:
        structure=f"{{some_path}}{NAME_PT_FOLDER}structure.gro",
        trajectory=f"{{some_path}}{NAME_PT_FOLDER}trajectory.xtc",
        structure1=f"{{some_path}}{NAME_PT_FOLDER}molecule1.gro",
        structure2=f"{{some_path}}{NAME_PT_FOLDER}molecule2.gro",
        translation_rotation_script = f"{PATH_VMD_SCRIPTS}{MOLECULE_NAMES}script{{view_i}}.log"
    output:
        vmdlog=f"{{some_path}}{NAME_VMD_OUTPUT}frame_{{frame_index}}_view{{view_i}}",
        frame_plot=f"{{some_path}}{NAME_FRAME_PLOTS}frame_{{frame_index}}_view{{view_i}}.tga"
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
    Plot one specific frame and save it to molecular_plots/
    """
    input:
        structure=f"{{some_path}}{NAME_PT_FOLDER}structure.gro",
        trajectory=f"{{some_path}}{NAME_PT_FOLDER}trajectory.xtc",
        structure1=f"{{some_path}}{NAME_PT_FOLDER}molecule1.gro",
        structure2=f"{{some_path}}{NAME_PT_FOLDER}molecule2.gro",
        translation_rotation_script = f"{PATH_VMD_SCRIPTS}{MOLECULE_NAMES}script{{view_i}}.log",
        indices= f"{{some_path}}{{subfolder}}lowest_{{N}}.txt"
    output:
        vmdlog=f"{{some_path}}{{subfolder}}lowest_{{N}}_view{{view_i}}",
        frame_plot=f"{{some_path}}{{subfolder}}lowest_{{N}}_view{{view_i}}.tga"
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
