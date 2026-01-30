import numpy as np
from plotly.tools import DEFAULT_PLOTLY_COLORS

from molgri.molecules.transitions import MSM
from workflow.helpers.io import read_object, write_object, read_from_mdrun, get_num_atoms


rule make_msm:
    input:
        assignments = f"<outputs_assignment>full_assignment.npy",
        grid_info= "<outputs_network>grid_info.yaml",
    output:
        msm= f"<outputs_transitions>{{tau}}/msm.npz",
    run:
        full_assignments = read_object(input.assignments)
        grid_info = read_object(input.grid_info)
        N_gridpoints = grid_info["N_total"]


        my_msm = MSM(full_assignments, N_gridpoints)
        transition_matrix = my_msm.get_one_tau_transition_matrix(int(wildcards.tau), noncorrelated_windows=False)
        write_object(transition_matrix, output.msm)

rule run_decomposition_msm:
    """
    As output we want to have eigenvalues, eigenvectors. Es input we get a (sparse) rate matrix.
    """
    input:
        msm = f"<outputs_transitions>{{tau}}/msm.npz"
    benchmark:
        f"<outputs_transitions>{{tau}}/timing_decomposition.txt"
    output:
        eigenvalues = f"<outputs_transitions>{{tau}}/eigenvalues.npy",
        eigenvectors = f"<outputs_transitions>{{tau}}/eigenvectors.npy"
    run:
        from molgri.molecules.transitions import DecompositionTool

        my_matrix = read_object(input.msm)

        # calculation
        dt = DecompositionTool(my_matrix)

        all_eigenval, all_eigenvec = dt.get_decomposition(tol=1e-5, maxiter=100000,
            which="LR",
            sigma=None)

        write_object(all_eigenval, output.eigenvalues)
        write_object(all_eigenvec,output.eigenvectors)

TAUS = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200]

rule get_implied_timescales:
    input:
        eigenvalues = f"<outputs_transitions>{{tau}}/eigenvalues.npy",
        runfile=f"<simulation>production.mdp"
    output:
        its = f"<outputs_transitions>{{tau}}/its.npy",
    run:
        eigenvalues = read_object(input.eigenvalues)
        eigenvalues = eigenvalues[1:]  # dropping the first one as it should be zero and cause issues

        writeout = int(read_from_mdrun(input.runfile,"nstxout-compressed"))
        time_step_ps = float(read_from_mdrun(input.runfile,"dt"))

        num_interesting_timescales = int(config["msm"]["num_interesting_timescales"])

        # save only the interesting ones
        while len(eigenvalues) < num_interesting_timescales:
            eigenvalues.append(np.nan)
        if len(eigenvalues) > 4:
            eigenvalues = eigenvalues[:4]

        its = - int(wildcards.tau) * writeout * time_step_ps / np.log(np.abs(eigenvalues))

        write_object(its, output.its)

rule find_indices_dominant_eigenvectors:
    """
    For each eigenvector find the structures that contribute the most to the eigenvector.
    """
    input:
        eigenvectors = f"<outputs_transitions>{{tau}}/eigenvectors.npy",
    output:
        abs_e_indices = expand(f"<outputs_indices>{{tau}}/0_eigenvector_{{j}}_largest_abs_values.txt",
            j=config["msm"]["num_extremes_to_plot"], allow_missing=True),
        pos_e_indices = expand(f"<outputs_indices>{{tau}}/{{i}}_eigenvector_{{j}}_most_positive.txt",
            i=range(1, config["msm"]["num_interesting_eigenvectors"]), j=config["msm"]["num_extremes_to_plot"],
            allow_missing=True),
        neg_e_indices= expand(f"<outputs_indices>{{tau}}/{{i}}_eigenvector_{{j}}_most_negative.txt",
            i=range(1,config["msm"]["num_interesting_eigenvectors"]),j=config["msm"]["num_extremes_to_plot"],
            allow_missing=True)
    run:
        from molgri.create_vmdlog import TrajectoryIndexingTool

        eigenvectors = read_object(input.eigenvectors)

        N_interesting_eigenvectors = config["msm"]["num_interesting_eigenvectors"]
        N_extremes_to_plot = config["msm"]["num_extremes_to_plot"]

        tit = TrajectoryIndexingTool()
        tit.set_eigenvectors(eigenvectors.T)

        # the shape of abs_e is (2*N_extremes_to_plot)
        # the shapes of pos_e and neg_e are (N_interesting_eigenvectors - 1, N_extremes_to_plot)
        abs_e, pos_e, neg_e = tit.get_all_dominant_structures(N_extremes_to_plot, N_interesting_eigenvectors)

        # save the absolute
        write_object(np.array(abs_e),output.abs_e_indices[0])

        # save the positive and negative indices
        for i in range(N_interesting_eigenvectors - 1):
            write_object(np.array(pos_e[i]), output.pos_e_indices[i])
            write_object(np.array(neg_e[i]),output.neg_e_indices[i])

rule plot_all_eigenvectors_as_lines:
    input:
        expand(f"<outputs_other_plots>eigenvectors_for_tau_{{tau}}.png", tau=TAUS)

rule plot_vmd_eigenvectors_as_lines:
    input:
        eigenvectors = f"<outputs_transitions>{{tau}}/eigenvectors.npy",
    output:
        plot = f"<outputs_other_plots>eigenvectors_for_tau_{{tau}}.png"
    run:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        eigenvector_array = read_object(input.eigenvectors)
        print(eigenvector_array.shape)

        N_interesting_eigenvectors = config["msm"]["num_interesting_eigenvectors"]

        fig = make_subplots(rows=N_interesting_eigenvectors,cols=1)

        for row in range(N_interesting_eigenvectors):
            print(eigenvector_array.shape[0], eigenvector_array[:, row].shape)
            fig.add_trace(
                go.Scatter(x=np.arange(eigenvector_array.shape[0]),y=eigenvector_array[:, row], line=dict(color="black"),
                    mode="lines"),row=1+row,col=1)

        fig.write_image(output.plot, scale=3)


rule plot_vmd_eigenvectors:
    input:
        structure = f"<simulation>structure.<ext_str>",
        structure1 = f"<simulation>molecule1.<ext_str>",
        trajectory = f"<outputs_assignment>wrapped_trajectory.<ext_trj>",
        abs_e_indices=rules.find_indices_dominant_eigenvectors.output.abs_e_indices,
        pos_e_indices=rules.find_indices_dominant_eigenvectors.output.pos_e_indices,
        neg_e_indices=rules.find_indices_dominant_eigenvectors.output.neg_e_indices,
        translation_rotation_script= f"<inputs_vmd>script3.log",
        grid_info = "<outputs_network>grid_info.yaml",
    output:
        vmdlog = f"<outputs_vmd>eigenvectors/{{tau}}/eigenvectors.log",
        plots = expand(f"<outputs_molecular_plots>eigenvectors/{{tau}}/eigenvector{{i}}.tga",
            i=range(config["msm"]["num_interesting_eigenvectors"]), allow_missing=True)
    run:
        from molgri.create_vmdlog import VMDCreator

        all_abs = np.array(read_object(input.abs_e_indices[0]))
        all_pos = []
        for pos_file in input.pos_e_indices:
            all_pos.append(read_object(pos_file))
        all_pos = np.array(all_pos)
        all_neg = []
        for neg_file in input.neg_e_indices:
            all_neg.append(read_object(neg_file))
        all_neg = np.array(all_neg)

        N_interesting_eigenvectors = config["msm"]["num_interesting_eigenvectors"]
        N_extremes_to_plot = config["msm"]["num_extremes_to_plot"]

        n1 = get_num_atoms(input.structure1)

        my_vmd = VMDCreator(f"index < {n1}", f"index >= {n1}")

        # drawing the rectangular box
        grid_info = read_object(input.grid_info)
        subgrid_limits = grid_info["subgrid_limits_A"]
        my_vmd.add_box(subgrid_limits[0][1], subgrid_limits[1][1], subgrid_limits[2][1])

        my_vmd.load_translation_rotation_script(input.translation_rotation_script)
        my_vmd.prepare_eigenvector_script(all_abs, all_pos, all_neg, plot_names=output.plots)
        my_vmd.write_text_to_file(output.vmdlog)

        shell("vmd  -dispdev text {input.structure} {input.trajectory} < {output.vmdlog}")

rule for_tau1:
    input:
        expand(f"<outputs_molecular_plots>eigenvectors/{{tau}}/eigenvector{{i}}.tga",i=range(config["msm"]["num_interesting_eigenvectors"]), tau=TAUS)


rule run_plot_its_msm:
    input:
        its = expand(f"<outputs_transitions>{{tau}}/its.npy", tau=TAUS),
        runfile=f"<simulation>production.mdp"
    output:
        plot_its = f"<outputs_other_plots>its.png"
    run:
        from plotly.subplots import make_subplots

        writeout = int(read_from_mdrun(input.runfile,"nstxout-compressed"))
        time_step_ps = float(read_from_mdrun(input.runfile,"dt"))

        xs = np.array(TAUS) * writeout * time_step_ps
        all_its = np.array([read_object(its_file) for its_file in input.its])


        fig = make_subplots(1, 2, shared_yaxes=False)
        for col in (1, 2):
            # gray triangle
            fig.add_scatter(x=[0, xs[-1], xs[-1]], y=[0, 0, xs[-1]], mode="lines", fill="toself", fillcolor="gray",
                                 line=dict(width=0), row=1, col=col)
            fig.update_layout(showlegend=False, xaxis_title=r"$\tau [ps]$", yaxis_title=r"ITS [ps]")
            fig.update_xaxes(title_text=r"$\tau [ps]$", row=1, col=col)
            fig.update_yaxes(title_text=r"ITS [ps]", row=1, col=col)
            # eigenvalues
            cols = DEFAULT_PLOTLY_COLORS

            for i, its in enumerate(all_its.T):
                if col==2:
                    xs = xs[:5]
                    its = its[:5]
                    fig.update_xaxes(range=[np.min(xs), np.max(xs)], row=1, col=col)
                fig.add_scatter(x=xs, y=its, mode="lines+markers", line=dict(width=2, color=cols[i]), row=1,
                                     col=col)
        fig.write_image(output.plot_its)