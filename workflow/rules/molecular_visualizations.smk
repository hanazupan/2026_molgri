"Perform analyses on energy-structure networks, eg identifying and plotting paths."
import matplotlib.pyplot as plt

from molgri.images.modifying_images import join_images

plt.switch_backend('agg')

pathvars:
    overlayed_vmd_frames = f'<outputs_molecular_plots>overlayed_vmd_frames/',
    stacked_vmd_frames = f'<outputs_molecular_plots>stacked_vmd_frames/',
    outputs_frame_plots = f'<outputs_molecular_plots>frames/',
    outputs_all_translations = f'<outputs_molecular_plots>all_translations/',
    outputs_all_rotations = f'<outputs_molecular_plots>all_rotations/',
    outputs_assignment_plots = f'<outputs_molecular_plots>compare_to_assignment/',

##################################### GENERAL FUNCTIONS ####################################################

def get_input_plot_frame(wc):
    if wc.simulation_or_pseudo == "simulation":
        path = "<simulation>"
    elif wc.simulation_or_pseudo == "pseudosimulation":
        path = "<pseudosimulation>"
    else:
        raise ValueError(f"Cannot understand this wildcard: {wc.simulation_or_pseudo}, should be simulation or pseudosimulation")

    if wc.both_or_m2 == "wrapped":
        trajectory_name = "<outputs_assignment>wrapped_trajectory"
    else:
        trajectory_name = f"{path}trajectory"

    print("Using trajectory ", trajectory_name)

    return {"structure": f"{path}structure.gro",
            "trajectory": f"{trajectory_name}.xtc",
            "structure1": f"{path}molecule1.gro",
            "grid_info": "<outputs_network>grid_info.yaml",
            "translation_rotation_script": f"<inputs_vmd>script{wc.view_i}.log"}

rule plot_one_frame:
    """
    Plot one specific frame and save it to molecular_plots/
    """
    input:
        unpack(get_input_plot_frame)
    output:
        vmdlog=f"<outputs_vmd>{{simulation_or_pseudo}}/frame_{{frame_index}}_view{{view_i}}_{{both_or_m2}}",
        frame_plot=f"<outputs_frame_plots>{{simulation_or_pseudo}}/frame_{{frame_index}}_view{{view_i}}_{{both_or_m2}}.tga"
    run:
        from molgri.create_vmdlog import VMDCreator
        from workflow.helpers.io import get_num_atoms, read_object

        n1 = get_num_atoms(input.structure1)

        my_vmd = VMDCreator(f"index < {n1}",f"index >= {n1}")
        my_vmd.load_translation_rotation_script(input.translation_rotation_script)

        # drawing the rectangular box
        grid_info = read_object(input.grid_info)
        subgrid_limits = grid_info["subgrid_limits_A"]
        my_vmd.add_box(subgrid_limits[0][1], subgrid_limits[1][1], subgrid_limits[2][1])

        index_to_plot = [int(wildcards.frame_index) + 1]

        # my_vmd.plot_multiple_overlappig_frames(index_to_plot,output.frame_plot,only_COM_of_m2=True)

        if wildcards.both_or_m2 == "m2":
            my_vmd.plot_m2_frames_individually(index_to_plot,[output.frame_plot])
        else:
            my_vmd.plot_frames_individually(index_to_plot,[output.frame_plot])


        my_vmd.write_text_to_file(output.vmdlog)


        shell("vmd  -dispdev text {input.structure} {input.trajectory} < {output.vmdlog}")

def get_input_plot_multiple_frames(wc):
    result = get_input_plot_frame(wc)
    result["indices"] = f"<outputs_indices>{wc.file_name}.txt"
    return result

rule plot_overlay_frames:
    """
    Plot multiple frames at the same time (eg. 10 lowest E structures, but this is a general rule).
    """
    input:
        unpack(get_input_plot_multiple_frames)
    output:
        vmdlog=f"<outputs_vmd>{{simulation_or_pseudo}}/{{file_name}}_view{{view_i}}_{{both_or_m2}}",
        frame_plot=f"<overlayed_vmd_frames>{{simulation_or_pseudo}}/{{file_name}}_view{{view_i}}_{{both_or_m2}}.tga"
    run:
        from molgri.create_vmdlog import VMDCreator
        from workflow.helpers.io import get_num_atoms, read_object

        indices = read_object(input.indices)

        n1 = get_num_atoms(input.structure1)

        my_vmd = VMDCreator(f"index < {n1}", f"index >= {n1}")
        my_vmd.load_translation_rotation_script(input.translation_rotation_script)

        index_to_plot = [int(i) + 1 for i in indices]

        if wildcards.both_or_m2=="COM":
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
    file_names = [f"<outputs_frame_plots>{wc.simulation_or_pseudo}/frame_{frame_index}_view3_{wc.both_or_m2}.tga" for frame_index in indices]
    return file_names


rule stack_vmd_frames:
    """
    In this case we don't want structures overlapped but plotted next to each other.
    """
    input:
        get_frame_indices
    output:
        joint_image = f"<stacked_vmd_frames>{{simulation_or_pseudo}}/{{file_name}}_{{both_or_m2}}.tga"
    shadow: "minimal"
    run:
        from molgri.images.modifying_images import trim_images_with_common_bbox, join_images
        modified_paths = [f"{os.path.split(file)[0]}/trimmed_{os.path.split(file)[1]}" for file in input]
        trim_images_with_common_bbox(input,modified_paths)
        join_images(modified_paths, output.joint_image, flip=False)

##################################### ALL TRANSLATIONS; ALL ROTATIONS ##################################################

rule plot_overlay_all_translations:
    """
    Show a VMD plot that overlaps all possible rotations of molecule2 at the first grid position.
    """
    input:
        frame_plot = f"<overlayed_vmd_frames>pseudosimulation/all_positions_first_rotation_view3_full.tga",
        frame_plot_COM = f"<overlayed_vmd_frames>pseudosimulation/all_positions_first_rotation_view3_COM.tga"

rule plot_overlay_all_rotations:
    """
    Show a VMD plot that overlaps all possible positions of molecule2 at the first orientation.
    """
    input:
        frame_plot= f"<overlayed_vmd_frames>pseudosimulation/all_rotations_first_position_view3_full.tga",
        frame_plot_COM= f"<overlayed_vmd_frames>pseudosimulation/all_rotations_first_position_view3_COM.tga"


rule stack_all_rotation_options:
    """
    Collect images of each rotation and plot them next to each other.
    """
    input:
        joint_image_both = "<stacked_vmd_frames>pseudosimulation/all_rotations_first_position_both.tga",
        joint_image_m2 = "<stacked_vmd_frames>pseudosimulation/all_rotations_first_position_m2.tga"

rule stack_all_translation_options:
    """
    Collect images of each translation and plot them next to each other. Warning, this can be a looot of subplots.
    """
    input:
        joint_image_both = "<stacked_vmd_frames>pseudosimulation/all_positions_first_rotation_both.tga",
        joint_image_m2 = "<stacked_vmd_frames>pseudosimulation/all_positions_first_rotation_m2.tga"

##################################### ASSIGNMENT ##################################################

# todo input function that finds the index of pseudotrajectory frame you wanna compare then the input only need to be these two frame plots

def get_input_trajectory_frame_i_find_pt_frame(wc):
    plot_trajectory_frame = f"<outputs_frame_plots>simulation/frame_{wc.i}_view3_wrapped.tga"

    # find the assignment_i
    all_assignments = read_object(f"{wc.path}assignment/full_assignment.npy")
    assignment_i = int(all_assignments[int(wc.i)])
    print(f"For simulation frame {int(wc.i)} I am plotting assigned frame {assignment_i}.")

    plot_pt_frame = f"<outputs_frame_plots>pseudosimulation/frame_{assignment_i}_view3_both.tga"
    return [plot_trajectory_frame, plot_pt_frame]


rule visually_test_full_assignment:
    input:
        get_input_trajectory_frame_i_find_pt_frame
    output:
        plot_comparison="{path}molecular_plots/compare_to_assignment/frame_{i}.tga"
    run:
        join_images(input, output.plot_comparison, flip=False)