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
        result["frame_gro"] = f"<outputs_assignment>trajectory_slices/frame_{wc.frame_index}.<ext_str>"
    elif wc.sim_pseudo_wrapped == "simulation":
        result["frame_gro"] = f"<simulation_traj_slices>frame_{wc.frame_index}.<ext_str>"
    elif wc.sim_pseudo_wrapped == "pseudosimulation":
        result["frame_gro"] = f"<pseudosimulation_traj_slices>frame_{wc.frame_index}.<ext_str>"
    elif wc.sim_pseudo_wrapped == "wrapped_COM":
        result["frame_gro"] = f"<outputs_assignment>trajectory_slices/m1_COM_m2_frame_{wc.frame_index}.<ext_str>"
        result["structure"] = f"<outputs_assignment>trajectory_slices/m1_COM_m2_frame_{wc.frame_index}.<ext_str>"
    elif wc.sim_pseudo_wrapped == "simulation_COM":
        result["frame_gro"] = f"<simulation_traj_slices>m1_COM_m2_frame_{wc.frame_index}.<ext_str>"
        result["structure"] = f"<simulation_traj_slices>m1_COM_m2_frame_{wc.frame_index}.<ext_str>"
    elif wc.sim_pseudo_wrapped == "pseudosimulation_COM":
        result["frame_gro"] = f"<pseudosimulation_traj_slices>m1_COM_m2_frame_{wc.frame_index}.<ext_str>"
        result["structure"] = f"<pseudosimulation_traj_slices>m1_COM_m2_frame_{wc.frame_index}.<ext_str>"
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
    elif "0_eigenvector" in wc.file_name:
        tau = int(wc.file_name.split("/")[0])
        indices_file = checkpoints.find_indices_dominant_eigenvectors.get(tau=tau).output.abs_e_indices
    indices = read_object(indices_file).astype(int)

    if wc.sim_pseudo_wrapped == "wrapped":
        result["all_frame_gros"] = tuple([f"<outputs_assignment>trajectory_slices/frame_{frame_index}.<ext_str>" for frame_index in indices])
    elif wc.sim_pseudo_wrapped == "simulation":
        result["all_frame_gros"] = tuple([f"<simulation_traj_slices>frame_{frame_index}.<ext_str>" for frame_index in indices])
    elif wc.sim_pseudo_wrapped == "pseudosimulation":
        result["all_frame_gros"] = tuple([f"<pseudosimulation_traj_slices>frame_{frame_index}.<ext_str>" for frame_index in indices])
    elif wc.sim_pseudo_wrapped == "wrapped_COM":
        result["all_frame_gros"] = tuple([f"<outputs_assignment>trajectory_slices/m1_COM_m2_frame_{frame_index}.<ext_str>" for frame_index in indices])
    elif wc.sim_pseudo_wrapped == "simulation_COM":
        result["all_frame_gros"] = tuple([f"<simulation_traj_slices>m1_COM_m2_frame_{frame_index}.<ext_str>" for frame_index in indices])
    elif wc.sim_pseudo_wrapped == "pseudosimulation_COM":
        result["all_frame_gros"] = tuple([f"<pseudosimulation_traj_slices>m1_COM_m2_frame_{frame_index}.<ext_str>" for frame_index in indices])
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

def input_red_blue(wc):
    result = input_base(wc)


    indices_pos_file = checkpoints.find_indices_dominant_eigenvectors.get(tau=wc.tau, i=wc.i).output.pos_e_indices
    indices_neg_file = checkpoints.find_indices_dominant_eigenvectors.get(tau=wc.tau,i=wc.i).output.neg_e_indices
    indices_pos = read_object(indices_pos_file).astype(int)
    indices_neg = read_object(indices_neg_file).astype(int)
    file_names_pos = tuple([f"<simulation>trajectory_slices/frame_{frame_index}.<ext_str>" for frame_index in indices_pos])
    file_names_neg = tuple([f"<simulation>trajectory_slices/frame_{frame_index}.<ext_str>" for frame_index in indices_neg])

    result["pos_e_structures"] = file_names_pos
    result["neg_e_structures"] = file_names_neg
    return result


rule new_vmd_plot_red_blue_overlapping_frames:
    input:
        unpack(input_red_blue)
    output:
        vmdlog=f"<outputs_vmd>simulation_eigenvectors/{{tau}}/{{i}}_eigenvector_zoom{{zoom_level}}_view{{view_index}}_{{com_or_full}}",
        frame_plot=f"<outputs_molecular_plots>simulation_eigenvectors/{{tau}}/{{i}}_eigenvector_zoom{{zoom_level}}_view{{view_index}}_{{com_or_full}}.tga",
        frame_plot_png = f"<outputs_molecular_plots>simulation_eigenvectors/{{tau}}/{{i}}_eigenvector_zoom{{zoom_level}}_view{{view_index}}_{{com_or_full}}.png.png"
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
        my_grid = read_object(input.grid)[:, :3]
        my_grid = my_grid[::N_rotations]

        if params.draw_gridpoints:
            gridpoints = my_grid
        else:
            gridpoints = None

        box_limits = [subgrid_limits[0][1],subgrid_limits[1][1],subgrid_limits[2][1]]

        my_vmd = VMDCreator(f"index < {n1}",f"index >= {n1}")

        my_vmd.prepare_eigenvector_script(num_red=len(input.pos_e_structures), num_blue=len(input.neg_e_structures),
            vmd_name=output.vmdlog, plot_name=output.frame_plot,
            box_limits=box_limits, draw_m1=params.draw_m1, draw_m2=params.draw_m2,
            draw_rectangular_box=params.draw_rectangular_box, gridpoints=gridpoints,
            zoom_level=int(wildcards.zoom_level), translation_rotation_script=input.translation_rotation_script)

        names_red = ' '.join(input.pos_e_structures)
        names_blue = ' '.join(input.neg_e_structures)
        shell("vmd  -dispdev text {input.structure} {names_red} {names_blue} < {output.vmdlog}")
        shell("convert {output.frame_plot} {output.frame_plot_png}")



def get_frame_indices(wc):
    if str(wc.file_name) == "all_rotations_first_position":
        indices_file = checkpoints.all_rotations_first_position_indices.get().output.indices
    elif str(wc.file_name) == "all_positions_first_rotation":
        indices_file = checkpoints.all_positions_first_rotation_indices.get().output.indices
    elif str(wc.file_name).startswith("lowest") and "pseudosimulation" in wc.sim_pseudo_wrapped:
        N = int(wc.file_name.split("_")[1])
        indices_file = checkpoints.lowest_E_indices_pseudosimulation.get(N=N).output.indices
    elif str(wc.file_name).startswith("lowest") and "pseudosimulation" not in wc.sim_pseudo_wrapped:
        N = int(wc.file_name.split("_")[1])
        indices_file = checkpoints.lowest_E_indices.get(N=N).output.indices
    elif "0_eigenvector" in wc.file_name:
        tau = int(wc.file_name.split("/")[0])
        indices_file = checkpoints.find_indices_dominant_eigenvectors.get(tau=tau).output.abs_e_indices
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
        frame_plot = f"<overlayed_vmd_frames>pseudosimulation/all_positions_first_rotation_zoom10_view1.png",
        frame_plot_COM = f"<overlayed_vmd_frames>pseudosimulation_COM/all_positions_first_rotation_zoom10_view1.png"

rule plot_overlay_all_rotations:
    """
    Show a VMD plot that overlaps all possible positions of molecule2 at the first orientation.
    """
    input:
        frame_plot= f"<overlayed_vmd_frames>pseudosimulation/all_rotations_first_position_zoom10_view1.png",
        frame_plot_COM= f"<overlayed_vmd_frames>pseudosimulation_COM/all_rotations_first_position_zoom10_view1.png"


rule stack_all_rotation_options:
    """
    Collect images of each rotation and plot them next to each other.
    """
    input:
        joint_image_both = "<stacked_vmd_frames>pseudosimulation/all_rotations_first_position_zoom10_view1.png",
        joint_image_m2 = "<stacked_vmd_frames>pseudosimulation_COM/all_rotations_first_position_zoom10_view1.png"

rule stack_all_translation_options:
    """
    Collect images of each translation and plot them next to each other. Warning, this can be a looot of subplots.
    """
    input:
        joint_image_both = "<stacked_vmd_frames>pseudosimulation/all_positions_first_rotation_zoom10_view1.png",
        joint_image_m2 = "<stacked_vmd_frames>pseudosimulation_COM/all_positions_first_rotation_zoom10_view1.png"

##################################### EIGENVECTORS ##################################################

rule get_zeroth_eigenvector_plot:
    """
    This is the usual overlayed plot.
    """
    input:
        expand(f"<overlayed_vmd_frames>{{tau}}/0_eigenvector_{{j}}_largest_abs_values_zoom10_view1.png",
            tau=config["msm"]["taus"], j=config["msm"]["num_extremes_to_plot"])

rule get_higher_eigenvector_plot:
    input:
        expand(f"<outputs_molecular_plots>simulation_eigenvectors/{{tau}}/{{i}}_eigenvector_zoom10_view1_{{com_or_full}}.png",
            tau=config["msm"]["taus"], i = range(1, int(config["msm"]["num_interesting_eigenvectors"])),
        com_or_full=["com", "full"])

##################################### ASSIGNMENT ##################################################

def get_input_trajectory_frame_i_find_pt_frame(wc):
    # find the assignment_i
    all_assignments = read_object(checkpoints.full_assignment.get().output.full_assignment)
    assignment_i = int(all_assignments[int(wc.i)])
    print(f"For simulation frame {int(wc.i)} I am plotting assigned frame {assignment_i}.")

    if "com" in wc.com_or_full:
        plot_trajectory_frame = f"<outputs_frame_plots>wrapped_COM/frame_{wc.i}_zoom{{zoom_level}}_view{{view_index}}.png"
        plot_pt_frame = f"<outputs_frame_plots>pseudosimulation_COM/frame_{assignment_i}_zoom{{zoom_level}}_view{{view_index}}.tga"
    else:
        plot_trajectory_frame = f"<outputs_frame_plots>wrapped/frame_{wc.i}_zoom{{zoom_level}}_view{{view_index}}.png"
        plot_pt_frame = f"<outputs_frame_plots>pseudosimulation/frame_{assignment_i}_zoom{{zoom_level}}_view{{view_index}}.tga"

    return [plot_trajectory_frame, plot_pt_frame]


rule visually_test_full_assignment:
    input:
        get_input_trajectory_frame_i_find_pt_frame
    output:
        plot_comparison="<outputs_molecular_plots>compare_to_assignment/frame_{i}_zoom{zoom_level}_view{view_index}_{com_or_full}.png"
    run:
        join_images(input, output.plot_comparison, flip=False)