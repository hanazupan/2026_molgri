import numpy as np
from plotly.tools import DEFAULT_PLOTLY_COLORS

from molgri.molecules.rate_merger import delete_rows_columns
from molgri.molecules.transitions import MSM
from workflow.helpers.io import read_object, write_object, read_from_mdrun


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

rule reduce_msm_size:
    input:
        msm= f"<outputs_transitions>{{tau}}/msm.npz",
    wildcard_constraints:
        tau= r"[1-9]\d*"
    output:
        reduced_msm= f"<outputs_transitions>{{tau}}/reduced_msm.npz",
        indices_to_keep = f"<outputs_transitions>{{tau}}/indices_to_keep.npy",
    run:
        msm = read_object(input.msm)
        reduced_msm, indices_to_keep = delete_rows_columns(msm,"msm")
        write_object(reduced_msm, output.reduced_msm)
        write_object(np.array(indices_to_keep),output.indices_to_keep)


rule run_decomposition_msm:
    """
    As output we want to have eigenvalues, eigenvectors. Es input we get a (sparse) rate matrix.
    """
    input:
        reduced_msm=f"<outputs_transitions>{{tau}}/reduced_msm.npz",
        indices_to_keep=f"<outputs_transitions>{{tau}}/indices_to_keep.npy",
        grid_info= "<outputs_network>grid_info.yaml",
    wildcard_constraints:
        tau= r"[1-9]\d*"
    benchmark:
        f"<outputs_transitions>{{tau}}/timing_decomposition.txt"
    output:
        eigenvalues = f"<outputs_transitions>{{tau}}/eigenvalues.npy",
        eigenvectors = f"<outputs_transitions>{{tau}}/eigenvectors.npy"
    run:
        from molgri.molecules.transitions import DecompositionTool

        grid_info = read_object(input.grid_info)
        total_length = int(grid_info["N_total"])
        kept_indices = read_object(input.indices_to_keep)
        my_matrix = read_object(input.reduced_msm)

        # calculation
        dt = DecompositionTool(my_matrix, kept_indices, total_length)
        all_eigenval, all_eigenvec = dt.decompose_msm()

        write_object(all_eigenval, output.eigenvalues)
        write_object(all_eigenvec,output.eigenvectors)



TAUS = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200]

rule get_implied_timescales:
    input:
        eigenvalues = f"<outputs_transitions>{{tau}}/eigenvalues.npy",
        runfile=f"<simulation>production.mdp"
    output:
        its = f"<outputs_transitions>{{tau}}/its.npy",
    params:
        N_eigenvec = config["eigenvectors"]["num_interesting_eigenvectors"]
    run:
        eigenvalues = read_object(input.eigenvalues)
        eigenvalues = eigenvalues[1:]  # dropping the first one as it should be zero and cause issues

        writeout = int(read_from_mdrun(input.runfile,"nstxout-compressed"))
        time_step_ps = float(read_from_mdrun(input.runfile,"dt"))

        num_interesting_timescales = int(params.N_eigenvec)

        # save only the interesting ones
        while len(eigenvalues) < num_interesting_timescales:
            eigenvalues.append(np.nan)
        if len(eigenvalues) > 4:
            eigenvalues = eigenvalues[:4]

        its = - int(wildcards.tau) * writeout * time_step_ps / np.log(np.abs(eigenvalues))

        write_object(its, output.its)



rule plot_all_eigenvectors_as_lines:
    input:
        expand(f"<outputs_other_plots>grouped_by_zcoo_eigenvectors_for_tau_{{tau}}.png", tau=[10])

rule plot_vmd_eigenvectors_as_lines_grouped_by_zcoo:
    input:
        eigenvectors = f"<outputs_transitions>{{tau}}/eigenvectors.npy",
        grid_info= rules.save_basic_grid_information.output.info_material
    output:
        plot = f"<outputs_other_plots>grouped_by_zcoo_eigenvectors_for_tau_{{tau}}.png"
    params:
        N_eigenvec = config["eigenvectors"]["num_interesting_eigenvectors"]
    run:
        import plotly.graph_objects as go
        import numpy as np
        from plotly.subplots import make_subplots

        grid_info = read_object(input.grid_info)
        N_rotations = grid_info["N_rotations"]
        N_translations = grid_info["N_translations"]
        subgrids = grid_info["subgrid_points"]
        len_x, len_y, len_z = len(subgrids[0]), len(subgrids[1]), len(subgrids[2])

        eigenvector_array = read_object(input.eigenvectors)
        N_interesting_eigenvectors = int(params.N_eigenvec)

        groups_by_rotation_index = np.repeat(np.arange(N_translations), N_rotations)
        groups_by_rotation_index = groups_by_rotation_index % len_z
        print(groups_by_rotation_index)
        print(groups_by_rotation_index[70:90])
        print(groups_by_rotation_index[1990:2010])

        fig = make_subplots(rows=N_interesting_eigenvectors,cols=1)

        for row in range(N_interesting_eigenvectors):
            eigenvector = eigenvector_array[:, row]
            out = np.bincount(groups_by_rotation_index ,weights=eigenvector,minlength=len_z)
            fig.add_trace(
                go.Bar(x=np.arange(len_z),y=out, text=[f"{i:>2}" for i in np.arange(len_z)]),row=1+row,col=1)
        fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
        #fig.update_yaxes(range=[-5, 5])
        #fig.update_yaxes(range=[-5, 0], row=1, col=1)
        fig.update_xaxes(showticklabels=False)
        fig.write_image(output.plot, scale=3)

rule plot_vmd_eigenvectors_as_lines_grouped_by_trans:
    input:
        eigenvectors = f"<outputs_transitions>{{tau}}/eigenvectors.npy",
        grid_info= rules.save_basic_grid_information.output.info_material
    output:
        plot = f"<outputs_other_plots>grouped_by_trans_eigenvectors_for_tau_{{tau}}.png"
    params:
        N_eigenvec = config["eigenvectors"]["num_interesting_eigenvectors"]
    run:
        import plotly.graph_objects as go
        import numpy as np
        from plotly.subplots import make_subplots

        grid_info = read_object(input.grid_info)
        N_rotations = grid_info["N_rotations"]
        N_translations = grid_info["N_translations"]
        subgrids = grid_info["subgrid_points"]
        len_x, len_y, len_z = len(subgrids[0]), len(subgrids[1]), len(subgrids[2])

        eigenvector_array = read_object(input.eigenvectors)
        N_interesting_eigenvectors = int(params.N_eigenvec)

        groups_by_rotation_index = np.repeat(np.arange(N_translations), N_rotations)
        print(groups_by_rotation_index)
        print(groups_by_rotation_index[70:90])

        fig = make_subplots(rows=N_interesting_eigenvectors,cols=1)

        for row in range(N_interesting_eigenvectors):
            eigenvector = eigenvector_array[:, row]
            out = np.bincount(groups_by_rotation_index ,weights=eigenvector,minlength=N_translations)
            fig.add_trace(
                go.Bar(x=np.arange(N_translations),y=out, text=[f"{i:>2}" for i in np.arange(N_translations)]),row=1+row,col=1)
        fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
        #fig.update_yaxes(range=[-5, 5])
        #fig.update_yaxes(range=[-5, 0], row=1, col=1)
        fig.update_xaxes(showticklabels=False)
        fig.write_image(output.plot, scale=3)

rule plot_vmd_eigenvectors_as_lines_grouped_by_rotation:
    input:
        eigenvectors = f"<outputs_transitions>{{tau}}/eigenvectors.npy",
        grid_info= rules.save_basic_grid_information.output.info_material
    output:
        plot = f"<outputs_other_plots>grouped_by_rotation_eigenvectors_for_tau_{{tau}}.png"
    params:
        N_eigenvec = config["eigenvectors"]["num_interesting_eigenvectors"]
    run:
        import plotly.graph_objects as go
        import numpy as np
        from plotly.subplots import make_subplots

        grid_info = read_object(input.grid_info)
        N_rotations = grid_info["N_rotations"]
        N_translations = grid_info["N_translations"]


        eigenvector_array = read_object(input.eigenvectors)
        N_interesting_eigenvectors = int(params.N_eigenvec)

        groups_by_rotation_index = np.tile(np.arange(N_rotations), N_translations)


        fig = make_subplots(rows=N_interesting_eigenvectors,cols=1)

        for row in range(N_interesting_eigenvectors):
            eigenvector = eigenvector_array[:, row]
            out = np.bincount(groups_by_rotation_index ,weights=eigenvector,minlength=N_rotations)
            fig.add_trace(
                go.Bar(x=np.arange(N_rotations),y=out, text=[f"{i:>2}" for i in np.arange(N_rotations)]),row=1+row,col=1)
        fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
        fig.update_yaxes(range=[-5, 5])
        fig.update_yaxes(range=[-5, 0], row=1, col=1)
        fig.update_xaxes(showticklabels=False)
        fig.write_image(output.plot, scale=3)


rule plot_vmd_eigenvectors_as_lines:
    input:
        eigenvectors = f"<outputs_transitions>{{tau}}/eigenvectors.npy",
    output:
        plot = f"<outputs_other_plots>eigenvectors_for_tau_{{tau}}.png"
    params:
        N_eigenvec = config["eigenvectors"]["num_interesting_eigenvectors"]
    run:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        eigenvector_array = read_object(input.eigenvectors)

        N_interesting_eigenvectors = int(params.N_eigenvec)

        fig = make_subplots(rows=N_interesting_eigenvectors,cols=1)

        for row in range(N_interesting_eigenvectors):
            fig.add_trace(
                go.Scatter(x=np.arange(eigenvector_array.shape[0]),y=eigenvector_array[:, row], line=dict(color="black"),
                    mode="lines"),row=1+row,col=1)

        fig.update_layout(showlegend=False, plot_bgcolor="white",)
        fig.write_image(output.plot, scale=3)

rule run_plot_its_msm:
    input:
        its = expand(f"<outputs_transitions>{{tau}}/its.npy", tau=config["msm"]["taus"]),
        runfile=f"<simulation>production.mdp"
    output:
        plot_its = f"<outputs_other_plots>its.png"
    run:
        from plotly.subplots import make_subplots

        writeout = int(read_from_mdrun(input.runfile,"nstxout-compressed"))
        time_step_ps = float(read_from_mdrun(input.runfile,"dt"))

        xs = np.array(config["msm"]["taus"]) * writeout * time_step_ps
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
                    xs = xs[:9]
                    its = its[:9]
                    fig.update_xaxes(range=[0, np.max(xs)], row=1, col=col)
                    fig.update_yaxes(range=[0, 20],row=1,col=col)
                fig.add_scatter(x=xs, y=its, mode="lines+markers", line=dict(width=2, color=cols[i]), row=1,
                                     col=col)
        fig.write_image(output.plot_its)