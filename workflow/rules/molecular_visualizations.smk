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
        vmdlog=f"<outputs_vmd>frame_{{frame_index}}_view{{view_i}}_{{both_or_m2}}",
        frame_plot=f"<outputs_frame_plots>frame_{{frame_index}}_view{{view_i}}_{{both_or_m2}}.tga"
    run:
        from molgri.create_vmdlog import VMDCreator
        from workflow.helpers.io import get_num_atoms

        n1 = get_num_atoms(input.structure1)
        n2 = get_num_atoms(input.structure2)

        my_vmd = VMDCreator(f"index < {n1}",f"index >= {n1}")
        my_vmd.load_translation_rotation_script(input.translation_rotation_script)
        index_to_plot = [int(wildcards.frame_index) + 1]

        if wildcards.both_or_m2 == "both":
            my_vmd.plot_frames_individually(index_to_plot,[output.frame_plot])
        elif wildcards.both_or_m2 == "m2":
            my_vmd.plot_m2_frames_individually(index_to_plot,[output.frame_plot])
        else:
            raise ValueError("Wilcard 'both_or_m2' must be either 'both' or 'm2'.")
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
        vmdlog=f"<outputs_vmd>{{file_name}}_view{{view_i}}_{{COM_or_not}}",
        frame_plot=f"<overlayed_vmd_frames>{{file_name}}_view{{view_i}}_{{COM_or_not}}.tga"
    run:
        from molgri.create_vmdlog import VMDCreator
        from workflow.helpers.io import get_num_atoms, read_object

        indices = read_object(input.indices)

        n1 = get_num_atoms(input.structure1)
        n2 = get_num_atoms(input.structure2)

        my_vmd = VMDCreator(f"index < {n1}", f"index >= {n1}")
        my_vmd.load_translation_rotation_script(input.translation_rotation_script)

        index_to_plot = [int(i) + 1 for i in indices]

        if wildcards.COM_or_not=="COM":
            plot_COM = True
        else:
            plot_COM = False

        my_vmd.plot_multiple_overlappig_frames(index_to_plot,output.frame_plot, only_COM_of_m2=plot_COM)
        my_vmd.write_text_to_file(output.vmdlog)

        shell("vmd  -dispdev text {input.structure} {input.trajectory} < {output.vmdlog}")

def get_frame_indices(wc):
    if str(wc.file_name) == "all_rotations_first_position":
        indices_file = checkpoints.all_rotations_first_position_indices.get().output.indices
    elif str(wc.file_name) == "all_positions_first_rotation":
        indices_file = checkpoints.all_positions_first_rotation_indices.get().output.indices
    indices = read_object(indices_file).astype(int)
    file_names = [f"<outputs_frame_plots>frame_{frame_index}_view2_{wc.both_or_m2}.tga" for frame_index in indices]
    return file_names


rule stack_vmd_frames:
    """
    In this case we don't want structures overlapped but plotted next to each other.
    """
    input:
        get_frame_indices
    output:
        joint_image = f"<stacked_vmd_frames>{{file_name}}_{{both_or_m2}}.png"
    shadow: "minimal"
    run:
        from molgri.images.modifying_images import trim_images_with_common_bbox, join_images
        modified_paths = [f"{os.path.split(file)[0]}/trimmed_{os.path.split(file)[1]}" for file in input]
        trim_images_with_common_bbox(input,modified_paths)
        join_images(modified_paths, output.joint_image, flip=False)