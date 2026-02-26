"Perform analyses on energy-structure networks, eg identifying and plotting paths."
import matplotlib.pyplot as plt

from molgri.create_vmdlog import VMDCreator
from molgri.images.modifying_images import trim_images_with_common_bbox, join_images
from workflow.helpers.find_right_input import where_to_look, find_the_right_frames, \
    find_the_right_structure, what_to_provide
from workflow.helpers.io import get_num_atoms, read_object

plt.switch_backend('agg')

pathvars:
    overlayed_vmd_frames = f'<outputs_molecular_plots>overlayed_vmd_frames/',
    stacked_vmd_frames = f'<outputs_molecular_plots>stacked_vmd_frames/',
    outputs_frame_plots = f'<outputs_molecular_plots>frames/',
    outputs_all_translations = f'<outputs_molecular_plots>all_translations/',
    outputs_all_rotations = f'<outputs_molecular_plots>all_rotations/',
    outputs_assignment_plots = f'<outputs_molecular_plots>compare_to_assignment/',

##################################### GENERAL FUNCTIONS ####################################################

def input_base(where, what, wc):
    structure_path = find_the_right_structure(where, what)
    return {"structure": structure_path,
            "structure1": "<pseudosimulation>molecule1.gro",
            "grid_info": "<outputs_network>grid_info.yaml",
            "translation_rotation_script": f"<inputs_vmd>script{wc.view_index}.log",
            "grid": "<outputs_network>grid.npy"}

def input_one_frame(wc):
    where = where_to_look(wc.sim_pseudo_wrapped)
    what = what_to_provide(wc.sim_pseudo_wrapped, for_a_structure=True)
    result = input_base(where, what, wc)
    what = what_to_provide(wc.sim_pseudo_wrapped,for_a_structure=False)
    frame = find_the_right_frames(where, what, [wc.frame_index])
    result["frame_gro"] = frame[0]
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

        n1 = get_num_atoms(input.structure1)
        grid_info = read_object(input.grid_info)
        subgrid_limits = grid_info["subgrid_limits_A"]
        N_rotations = int(grid_info["N_rotations"])

        if params.draw_gridpoints:
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


def collect_box_information(input):
    grid_info = read_object(input.grid_info)
    subgrid_limits = grid_info["subgrid_limits_A"]
    N_rotations = int(grid_info["N_rotations"])
    my_grid = read_object(input.grid)[:, :3]
    my_grid = my_grid[::N_rotations]

    box_limits = [subgrid_limits[0][1], subgrid_limits[1][1], subgrid_limits[2][1]]
    return box_limits, my_grid


# rule stack_vmd_frames:
#     """
#     In this case we don't want structures overlapped but plotted next to each other.
#     """
#     input:
#         get_frame_indices
#     output:
#         joint_image = f"<stacked_vmd_frames>{{sim_pseudo_wrapped}}/{{file_name}}_zoom{{zoom_level}}_view{{view_index}}.png"
#     shadow: "minimal"
#     run:
#         modified_paths = [f"{os.path.split(file)[0]}/trimmed_{os.path.split(file)[1]}" for file in input]
#         trim_images_with_common_bbox(input,modified_paths)
#         join_images(modified_paths, output.joint_image, flip=False)


# here follow specific versions of overlapped or stacked plots.´

##################################### LOWEST ENERGY ##################################################


def input_lowestE(wc):
    where = where_to_look(wc.sim_pseudo_wrapped)
    what = what_to_provide(wc.sim_pseudo_wrapped, for_a_structure=True)
    result = input_base(where, what, wc)
    indices_file = checkpoints.lowest_E_indices.get(path=where, N=wc.N).output.indices

    what = what_to_provide(wc.sim_pseudo_wrapped,for_a_structure=False)
    indices = read_object(indices_file).astype(int)
    result["all_frame_gros"] = find_the_right_frames(where, what, indices)
    return result

rule lowestE_overlapping_frames:
    input:
        unpack(input_lowestE)
    output:
        vmdlog=f"<outputs_vmd>{{sim_pseudo_wrapped}}/{{N}}_lowest_zoom{{zoom_level}}_view{{view_index}}",
        frame_plot=f"<outputs_molecular_plots>lowest_energy/{{sim_pseudo_wrapped}}/{{N}}_lowest_zoom{{zoom_level}}_view{{view_index}}.tga",
        frame_plot_png = f"<outputs_molecular_plots>lowest_energy/{{sim_pseudo_wrapped}}/{{N}}_lowest_zoom{{zoom_level}}_view{{view_index}}.png"
    run:
        n1 = get_num_atoms(input.structure1)
        box_limits, gridpoints = collect_box_information(input)

        my_vmd = VMDCreator(f"index < {n1}",f"index >= {n1}")


        my_vmd.prepare_frame_script(vmd_name=output.vmdlog, plot_name=output.frame_plot,
            num_frames=len(input.all_frame_gros),
            box_limits=box_limits, draw_m1=True, draw_m2=True,
            draw_rectangular_box=False, gridpoints=None,
            zoom_level=int(wildcards.zoom_level), translation_rotation_script=input.translation_rotation_script)

        names_all_frames = ' '.join(input.all_frame_gros)

        shell("vmd  -dispdev text {input.structure} {names_all_frames} < {output.vmdlog}")
        shell("convert {output.frame_plot} {output.frame_plot_png}")

##################################### ALL TRANSLATIONS; ALL ROTATIONS ##################################################

def input_all_translations(wc):
    where = "<pseudosimulation>"
    what = what_to_provide(wc.COM_or_full, for_a_structure=True)
    result = input_base(where, what, wc)

    indices_file = checkpoints.all_positions_first_rotation_indices.get().output.indices
    indices = read_object(indices_file).astype(int)
    what = what_to_provide(wc.COM_or_full,for_a_structure=False)
    result["all_frame_gros"] = find_the_right_frames("<pseudosimulation>", what, indices)
    return result

rule all_translations_overlapping_frames:
    input:
        unpack(input_all_translations)
    output:
        vmdlog=f"<outputs_vmd>overlapping_all_translations_zoom{{zoom_level}}_view{{view_index}}_{{COM_or_full}}",
        frame_plot=f"<outputs_molecular_plots>all_translations/zoom{{zoom_level}}_view{{view_index}}_{{COM_or_full}}.tga",
        frame_plot_png = f"<outputs_molecular_plots>all_translations/zoom{{zoom_level}}_view{{view_index}}_{{COM_or_full}}.png"
    run:
        n1 = get_num_atoms(input.structure1)
        box_limits, gridpoints = collect_box_information(input)

        my_vmd = VMDCreator(f"index < {n1}",f"index >= {n1}")


        my_vmd.prepare_frame_script(vmd_name=output.vmdlog, plot_name=output.frame_plot,
            num_frames=len(input.all_frame_gros),
            box_limits=box_limits, draw_m1=True, draw_m2=True,
            draw_rectangular_box=False, gridpoints=None,
            zoom_level=int(wildcards.zoom_level), translation_rotation_script=input.translation_rotation_script)

        names_all_frames = ' '.join(input.all_frame_gros)

        shell("vmd  -dispdev text {input.structure} {names_all_frames} < {output.vmdlog}")
        shell("convert {output.frame_plot} {output.frame_plot_png}")


def input_all_rotations(wc):
    where = "<pseudosimulation>"
    what = what_to_provide(wc.COM_or_full, for_a_structure=True)
    result = input_base(where, what, wc)

    indices_file = checkpoints.all_rotations_first_position_indices.get().output.indices
    indices = read_object(indices_file).astype(int)
    what = what_to_provide(wc.COM_or_full,for_a_structure=False)
    result["all_frame_gros"] = find_the_right_frames("<pseudosimulation>", what, indices)
    return result

rule all_rotations_overlapping_frames:
    input:
        unpack(input_all_rotations)
    output:
        vmdlog=f"<outputs_vmd>overlapping_all_rotations_zoom{{zoom_level}}_view{{view_index}}_{{COM_or_full}}",
        frame_plot=f"<outputs_molecular_plots>all_rotations/zoom{{zoom_level}}_view{{view_index}}_{{COM_or_full}}.tga",
        frame_plot_png = f"<outputs_molecular_plots>all_rotations/zoom{{zoom_level}}_view{{view_index}}_{{COM_or_full}}.png"
    run:
        n1 = get_num_atoms(input.structure1)
        box_limits, gridpoints = collect_box_information(input)

        my_vmd = VMDCreator(f"index < {n1}",f"index >= {n1}")


        my_vmd.prepare_frame_script(vmd_name=output.vmdlog, plot_name=output.frame_plot,
            num_frames=len(input.all_frame_gros),
            box_limits=box_limits, draw_m1=True, draw_m2=True,
            draw_rectangular_box=False, gridpoints=None,
            zoom_level=int(wildcards.zoom_level), translation_rotation_script=input.translation_rotation_script)

        names_all_frames = ' '.join(input.all_frame_gros)

        shell("vmd  -dispdev text {input.structure} {names_all_frames} < {output.vmdlog}")
        shell("convert {output.frame_plot} {output.frame_plot_png}")


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

rule stack_all_eigenvectors:
    input:
        all_eigenvectors_one_tau = expand(f"<outputs_molecular_plots>eigenvectors/{{tau}}/{{i}}th_eigenvector_zoom{{zoom_level}}_view{{view_index}}_{{COM_or_full}}.png",
            i=range(config["msm"]["num_interesting_eigenvectors"]), allow_missing=True),
    output:
        all_eigenvectors_one_tau = f"<outputs_molecular_plots>eigenvectors/{{tau}}/ALL_EIGENVECTORS_zoom{{zoom_level}}_view{{view_index}}_{{COM_or_full}}.png"
    run:
        join_images(input,output.all_eigenvectors_one_tau,flip=False)

rule get_all_eigenvectors:
    input:
        expand(f"<outputs_molecular_plots>eigenvectors/{{tau}}/ALL_EIGENVECTORS_zoom{{zoom_level}}_view{{view_index}}_{{COM_or_full}}.png",
            tau=config["msm"]["taus"], zoom_level=[8],view_index=[1, 4],
            COM_or_full=["full", "COM"]),

def input_zeroth_eigenvector(wc):
    where = "<outputs_assignment>"
    what = what_to_provide(wc.COM_or_full, for_a_structure=True)
    result = input_base(where, what, wc)


    indices_file = checkpoints.find_indices_dominant_eigenvectors.get(tau=wc.tau).output.abs_e_indices
    indices_file = indices_file[0]
    indices = read_object(indices_file).astype(int)
    what = what_to_provide(wc.COM_or_full,for_a_structure=False)
    result["all_frame_gros"] = find_the_right_frames(where, what, indices)

    return result

rule zeroth_eigenvector_overlapping_frames:
    input:
        unpack(input_zeroth_eigenvector)
    output:
        vmdlog=f"<outputs_vmd>eigenvectors/{{tau}}/0th_eigenvector_zoom{{zoom_level}}_view{{view_index}}_{{COM_or_full}}",
        frame_plot=f"<outputs_molecular_plots>eigenvectors/{{tau}}/0th_eigenvector_zoom{{zoom_level}}_view{{view_index}}_{{COM_or_full}}.tga",
        frame_plot_png = f"<outputs_molecular_plots>eigenvectors/{{tau}}/0th_eigenvector_zoom{{zoom_level}}_view{{view_index}}_{{COM_or_full}}.png"
    run:
        n1 = get_num_atoms(input.structure1)
        box_limits, gridpoints = collect_box_information(input)

        my_vmd = VMDCreator(f"index < {n1}",f"index >= {n1}")


        my_vmd.prepare_frame_script(vmd_name=output.vmdlog, plot_name=output.frame_plot,
            num_frames=len(input.all_frame_gros),
            box_limits=box_limits, draw_m1=True, draw_m2=True,
            draw_rectangular_box=False, gridpoints=None,
            zoom_level=int(wildcards.zoom_level), translation_rotation_script=input.translation_rotation_script)

        names_all_frames = ' '.join(input.all_frame_gros)

        shell("vmd  -dispdev text {input.structure} {names_all_frames} < {output.vmdlog}")
        shell("convert {output.frame_plot} {output.frame_plot_png}")


def input_red_blue(wc):
    where = "<outputs_assignment>"
    what = what_to_provide(wc.COM_or_full, for_a_structure=True)
    result = input_base(where, what, wc)

    indices_pos_file = checkpoints.find_indices_dominant_eigenvectors.get(tau=wc.tau, i=wc.i).output.pos_e_indices
    indices_neg_file = checkpoints.find_indices_dominant_eigenvectors.get(tau=wc.tau,i=wc.i).output.neg_e_indices
    indices_pos = read_object(indices_pos_file[0]).astype(int)
    indices_neg = read_object(indices_neg_file[0]).astype(int)

    what = what_to_provide(wc.COM_or_full,for_a_structure=False)
    result["pos_e_structures"] = find_the_right_frames(where, what, indices_pos)
    result["neg_e_structures"] = find_the_right_frames(where, what, indices_neg)
    return result


rule higher_eigenvector_overlapping_frames:
    input:
        unpack(input_red_blue)
    wildcard_constraints:
        i = r"[1-9]\d*"
    output:
        vmdlog=f"<outputs_vmd>eigenvectors/{{tau}}/{{i}}th_eigenvector_zoom{{zoom_level}}_view{{view_index}}_{{COM_or_full}}",
        frame_plot=f"<outputs_molecular_plots>eigenvectors/{{tau}}/{{i}}th_eigenvector_zoom{{zoom_level}}_view{{view_index}}_{{COM_or_full}}.tga",
        frame_plot_png = f"<outputs_molecular_plots>eigenvectors/{{tau}}/{{i}}th_eigenvector_zoom{{zoom_level}}_view{{view_index}}_{{COM_or_full}}.png"
    run:
        from molgri.create_vmdlog import VMDCreator
        from workflow.helpers.io import get_num_atoms, read_object

        n1 = get_num_atoms(input.structure1)
        box_limits, gridpoints = collect_box_information(input)


        my_vmd = VMDCreator(f"index < {n1}",f"index >= {n1}")

        my_vmd.prepare_eigenvector_script(num_red=len(input.pos_e_structures), num_blue=len(input.neg_e_structures),
            vmd_name=output.vmdlog, plot_name=output.frame_plot,
            box_limits=box_limits, draw_rectangular_box=False, gridpoints=None,
            zoom_level=int(wildcards.zoom_level), translation_rotation_script=input.translation_rotation_script)

        names_red = ' '.join(input.pos_e_structures)
        names_blue = ' '.join(input.neg_e_structures)
        print(f"vmd {input.structure} {names_red} {names_blue} ")
        shell("vmd  -dispdev text {input.structure} {names_red} {names_blue} < {output.vmdlog}")
        shell("convert {output.frame_plot} {output.frame_plot_png}")
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


def input_assignment_overlapping(wc):
    what = what_to_provide(wc.COM_or_full, for_a_structure=True)
    result = input_base("<pseudosimulation>", what, wc)

    all_assignments = read_object(checkpoints.full_assignment.get().output.full_assignment)
    assignment_i = int(all_assignments[int(wc.i)])

    what = what_to_provide(wc.COM_or_full,for_a_structure=False)
    structure_1 = find_the_right_frames("<outputs_assignment>", what, [int(wc.i)])[0]
    structure_2 = find_the_right_frames("<pseudosimulation>",what,[assignment_i])[0]
    result["all_frame_gros"] = tuple([structure_1, structure_2])
    return result

rule assignment_overlapping:
    input:
        unpack(input_assignment_overlapping)
    output:
        vmdlog=f"<outputs_vmd>compare_to_assignment/frame_{{i}}_zoom{{zoom_level}}_view{{view_index}}_{{COM_or_full}}",
        frame_plot=f"<outputs_molecular_plots>compare_to_assignment/overlapping_frame_{{i}}_zoom{{zoom_level}}_view{{view_index}}_{{COM_or_full}}.tga",
        frame_plot_png = f"<outputs_molecular_plots>compare_to_assignment/overlapping_frame_{{i}}zoom{{zoom_level}}_view{{view_index}}_{{COM_or_full}}.png"
    run:
        n1 = get_num_atoms(input.structure1)
        box_limits, gridpoints = collect_box_information(input)

        my_vmd = VMDCreator(f"index < {n1}",f"index >= {n1}")


        my_vmd.prepare_frame_script(vmd_name=output.vmdlog, plot_name=output.frame_plot,
            num_frames=len(input.all_frame_gros),
            box_limits=box_limits, draw_m1=True, draw_m2=True,
            draw_rectangular_box=True, gridpoints=gridpoints,
            zoom_level=int(wildcards.zoom_level), translation_rotation_script=input.translation_rotation_script)

        names_all_frames = ' '.join(input.all_frame_gros)

        shell("vmd  -dispdev text {input.structure} {names_all_frames} < {output.vmdlog}")
        shell("convert {output.frame_plot} {output.frame_plot_png}")

rule visually_test_full_assignment:
    input:
        get_input_trajectory_frame_i_find_pt_frame
    output:
        plot_comparison="<outputs_molecular_plots>compare_to_assignment/frame_{i}_zoom{zoom_level}_view{view_index}_{com_or_full}.png"
    run:
        join_images(input, output.plot_comparison, flip=False)