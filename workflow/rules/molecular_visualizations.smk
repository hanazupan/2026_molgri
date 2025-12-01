"Perform analyses on energy-structure networks, eg identifying and plotting paths."
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('agg')


rule all:
    input:
        expand("/home/hanaz63/2026_molgri/outputs/pseudotrajectories/graphene_xylene/graphene_grid_D1/molecular_plots/frame_{i}.tga",
            i=[1, 5, 20])


rule plot_one_frame:
    """
    Plot one specific frame and save it to molecular_plots/
    """
    input:
        structure=f"{{where}}structure.gro",
        trajectory=f"{{where}}trajectory.xtc",
        structure1=f"{{where}}m1.gro",
        structure2=f"{{where}}m2.gro",
    output:
        vmdlog="{where}molecular_vmdlog/frame_{frame_index}",
        frame_plot="{where}molecular_plots/frame_{frame_index}.tga"
    run:
        from molgri.create_vmdlog import VMDCreator
        from workflow.helpers.io import get_num_atoms

        n1 = get_num_atoms(input.structure1)
        n2 = get_num_atoms(input.structure2)


        my_vmd = VMDCreator(str(n1), str(n2))
        #my_vmd.load_translation_rotation_script(my_script)

        index_to_plot = [int(wildcards.frame_index) + 1]

        my_vmd.plot_these_structures(index_to_plot,[output.frame_plot])
        my_vmd.write_text_to_file(output.vmdlog)

        shell("vmd  -dispdev text {input.structure} {input.trajectory} < {output.vmdlog}")




# rule join_plots_lowestE:
#     input:
#         prepare_all_lowestE_plots
#     output:
#         joint_plot = "{where}lowest_energy/all_lowestE.png"
#     run:
#         from molgri.plotting.modifying_images import trim_images_with_common_bbox, join_images
#         modified_paths = [f"{os.path.split(file)[0]}/trimmed_{os.path.split(file)[1]}" for file in input]
#         trim_images_with_common_bbox(input,modified_paths)
#         join_images(modified_paths, output.joint_plot)
#
# rule collect_these_images:
#     """
#     Over different sub-folders (eg different cut-offs) collect the same image eg. first eigenvector.
#     """
#     input:
#         all_images = expand("{where}absolute_lim_{limit}/eigenvectors/{what}.png", limit=["1", "3", "5", "10", "20", "50", "100", "200", "500", "1000"], allow_missing=True)
#     output:
#         joint_image = "{where}joint_images/{what}.png"
#     run:
#         from molgri.plotting.modifying_images import trim_images_with_common_bbox, join_images
#         modified_paths = [f"{os.path.split(file)[0]}/trimmed_{os.path.split(file)[1]}" for file in input]
#         trim_images_with_common_bbox(input,modified_paths)
#         join_images(modified_paths, output.joint_image, flip=False)

