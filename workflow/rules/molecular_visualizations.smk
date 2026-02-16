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

def input_base(wc):
    return {"structure": "<pseudosimulation>structure.gro",
            "structure1": "<pseudosimulation>molecule1.gro",
            "grid_info": "<outputs_network>grid_info.yaml",
            "translation_rotation_script": f"<inputs_vmd>script{wc.view_index}.log",
            "grid": "<outputs_network>grid.npy"}

def input_one_frame(wc):
    result = input_base(wc)
    if wc.sim_pseudo_wrapped == "wrapped":
        pass # TODO
    elif wc.sim_pseudo_wrapped == "simulation":
        result["frame_gro"] = f"<simulation_traj_slices>frame_{wc.frame_index}.<ext_str>"
    elif wc.sim_pseudo_wrapped == "pseudosimulation":
        result["frame_gro"] = f"<pseudosimulation_traj_slices>frame_{wc.frame_index}.<ext_str>"
    else:
        raise ValueError("Folder must be 'wrapped', 'simulation' or 'pseudosimulation'.")
    return result

rule new_vmd_plot_one_frame:
    input:
        unpack(input_one_frame)
    output:
        vmdlog=f"<outputs_vmd>{{sim_pseudo_wrapped}}/frame_{{frame_index}}_zoom{{zoom_level}}_view{{view_index}}",
        frame_plot=f"<outputs_frame_plots>{{sim_pseudo_wrapped}}/frame_{{frame_index}}_zoom{{zoom_level}}_view{{view_index}}.tga",
        frame_plot_png = f"<outputs_frame_plots>{{sim_pseudo_wrapped}}/frame_{{frame_index}}_zoom{{zoom_level}}_view{{view_index}}.png"
    params:
        draw_m1 = True,
        draw_m2 = True,
        draw_rectangular_box = True,
        draw_gridpoints = True,
        center_on_box = True
    run:
        from molgri.create_vmdlog import VMDCreator
        from workflow.helpers.io import get_num_atoms, read_object

        n1 = get_num_atoms(input.structure1)
        grid_info = read_object(input.grid_info)
        subgrid_limits = grid_info["subgrid_limits_A"]
        N_rotations = int(grid_info["N_rotations"])


        if params.draw_gridpoints:
            #print()
            my_grid = read_object(input.grid)[:, :3]
            my_grid = my_grid[::N_rotations]
            gridpoints = my_grid
        else:
            gridpoints = None

        box_limits = [subgrid_limits[0][1],subgrid_limits[1][1],subgrid_limits[2][1]]

        my_vmd = VMDCreator(f"index < {n1}",f"index >= {n1}")

        my_vmd.prepare_frame_script(vmd_name=output.vmdlog, plot_name=output.frame_plot, num_frames=1,
            box_limits=box_limits, draw_m1=params.draw_m1, draw_m2=params.draw_m2,
            draw_rectangular_box=params.draw_rectangular_box, gridpoints=gridpoints,
            zoom_level=int(wildcards.zoom_level), translation_rotation_script=input.translation_rotation_script)

        shell("vmd  -dispdev text {input.structure} {input.frame_gro} < {output.vmdlog}")
        shell("convert {output.frame_plot} {output.frame_plot_png}")


def input_multiple_overlapping_frames(wc):
    result = input_base(wc)
    if str(wc.file_name) == "all_rotations_first_position":
        indices_file = checkpoints.all_rotations_first_position_indices.get().output.indices
    elif str(wc.file_name) == "all_positions_first_rotation":
        indices_file = checkpoints.all_positions_first_rotation_indices.get().output.indices
    elif str(wc.file_name).startswith("lowest") and wc.sim_pseudo_wrapped == "pseudosimulation":
        N = int(wc.file_name.split("_")[1])
        indices_file = checkpoints.lowest_E_indices_pseudosimulation.get(N=N).output.indices
    elif str(wc.file_name).startswith("lowest") and wc.sim_pseudo_wrapped != "pseudosimulation":
        N = int(wc.file_name.split("_")[1])
        indices_file = checkpoints.lowest_E_indices.get(N=N).output.indices
    indices = read_object(indices_file).astype(int)

    if wc.sim_pseudo_wrapped == "wrapped":
        pass # TODO
    elif wc.sim_pseudo_wrapped == "simulation":
        result["all_frame_gros"] = tuple([f"<simulation_traj_slices>frame_{frame_index}.<ext_str>" for frame_index in indices])
    elif wc.sim_pseudo_wrapped == "pseudosimulation":
        result["all_frame_gros"] = tuple([f"<pseudosimulation_traj_slices>frame_{frame_index}.<ext_str>" for frame_index in indices])
    else:
        raise ValueError("Folder must be 'wrapped', 'simulation' or 'pseudosimulation'.")
    return result

rule new_vmd_plot_multiple_overlapping_frames:
    input:
        unpack(input_multiple_overlapping_frames)
    output:
        vmdlog=f"<outputs_vmd>{{sim_pseudo_wrapped}}/{{file_name}}_zoom{{zoom_level}}_view{{view_index}}",
        frame_plot=f"<overlayed_vmd_frames>{{sim_pseudo_wrapped}}/{{file_name}}_zoom{{zoom_level}}_view{{view_index}}.tga",
        frame_plot_png = f"<overlayed_vmd_frames>{{sim_pseudo_wrapped}}/{{file_name}}_zoom{{zoom_level}}_view{{view_index}}.png"
    params:
        draw_m1 = True,
        draw_m2 = True,
        draw_rectangular_box = True,
        draw_gridpoints = True,
        center_on_box = True
    run:
        print(input)
        from molgri.create_vmdlog import VMDCreator
        from workflow.helpers.io import get_num_atoms, read_object

        n1 = get_num_atoms(input.structure1)
        grid_info = read_object(input.grid_info)
        subgrid_limits = grid_info["subgrid_limits_A"]
        N_rotations = int(grid_info["N_rotations"])
        my_grid = read_object(input.grid)[:, :3]
        my_grid = my_grid[::N_rotations]

        if params.draw_gridpoints:
            gridpoints = my_grid
        else:
            gridpoints = None

        box_limits = [subgrid_limits[0][1],subgrid_limits[1][1],subgrid_limits[2][1]]

        my_vmd = VMDCreator(f"index < {n1}",f"index >= {n1}")

        my_vmd.prepare_frame_script(vmd_name=output.vmdlog, plot_name=output.frame_plot,
            num_frames=len(input.all_frame_gros),
            box_limits=box_limits, draw_m1=params.draw_m1, draw_m2=params.draw_m2,
            draw_rectangular_box=params.draw_rectangular_box, gridpoints=gridpoints,
            zoom_level=int(wildcards.zoom_level), translation_rotation_script=input.translation_rotation_script)

        names_all_frames = ' '.join(input.all_frame_gros)
        shell("vmd  -dispdev text {input.structure} {names_all_frames} < {output.vmdlog}")
        shell("convert {output.frame_plot} {output.frame_plot_png}")



def get_frame_indices(wc):
    if str(wc.file_name) == "all_rotations_first_position":
        indices_file = checkpoints.all_rotations_first_position_indices.get().output.indices
    elif str(wc.file_name) == "all_positions_first_rotation":
        indices_file = checkpoints.all_positions_first_rotation_indices.get().output.indices
    elif str(wc.file_name).startswith("lowest") and wc.sim_pseudo_wrapped == "pseudosimulation":
        N = int(wc.file_name.split("_")[1])
        indices_file = checkpoints.lowest_E_indices_pseudosimulation.get(N=N).output.indices
    elif str(wc.file_name).startswith("lowest") and wc.sim_pseudo_wrapped != "pseudosimulation":
        N = int(wc.file_name.split("_")[1])
        indices_file = checkpoints.lowest_E_indices.get(N=N).output.indices
    indices = read_object(indices_file).astype(int)
    file_names = [f"<outputs_frame_plots>{wc.sim_pseudo_wrapped}/frame_{frame_index}_zoom{wc.zoom_level}_view{wc.view_index}.png" for frame_index in indices]
    return file_names


rule stack_vmd_frames:
    """
    In this case we don't want structures overlapped but plotted next to each other.
    """
    input:
        get_frame_indices
    output:
        joint_image = f"<stacked_vmd_frames>{{sim_pseudo_wrapped}}/{{file_name}}_zoom{{zoom_level}}_view{{view_index}}.png"
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
    all_assignments = read_object(checkpoints.full_assignment.get().output.full_assignment) #f"{wc.path}assignment/full_assignment.npy"
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