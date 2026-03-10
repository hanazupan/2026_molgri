import numpy as np

from molgri.images.plotting import show_array
from molgri.molecules.rate_merger import delete_rows_columns
from molgri.molecules.transitions import SQRA
from workflow.helpers.io import read_object, write_object

rule make_sqra:
    input:
        energies = "<pseudosimulation>energy.csv",
        volumes = "<outputs_network>volumes.npy",
        distances= "<outputs_network>distances.npz",
        surfaces= "<outputs_network>surfaces.npz"
    output:
        rate_matrix = f"<outputs_transitions>sqra/sqra.npz",
    params:
        T_in_K = 293,
        diffusion_coefficient = 1,
    run:
        my_energy = read_object(input.energies)
        my_energy_array = my_energy["Energy [kJ/mol]"].to_numpy()
        volumes = read_object(input.volumes)
        distances = read_object(input.distances)
        surfaces = read_object(input.surfaces)

        sqra = SQRA(energies=my_energy_array,volumes=volumes,distances=distances,surfaces=surfaces)

        rate_matrix = sqra.get_rate_matrix(params.diffusion_coefficient,params.T_in_K)
        # saving to file
        write_object(rate_matrix, output.rate_matrix)



rule reduce_sqra_size:
    input:
        sqra= f"<outputs_transitions>sqra/sqra.npz",
    output:
        reduced_sqra= f"<outputs_transitions>sqra/reduced_sqra.npz",
        indices_to_keep = f"<outputs_transitions>sqra/indices_to_keep.npy",
    run:
        sqra = read_object(input.sqra)
        reduced_sqra, indices_to_keep = delete_rows_columns(sqra,"sqra")
        write_object(reduced_sqra, output.reduced_sqra)
        write_object(np.array(indices_to_keep),output.indices_to_keep)

rule run_decomposition_sqra:
    """
    As output we want to have eigenvalues, eigenvectors. Es input we get a (sparse) rate matrix.
    """
    input:
        reduced_msm=f"<outputs_transitions>sqra/reduced_sqra.npz",
        indices_to_keep=f"<outputs_transitions>sqra/indices_to_keep.npy",
        grid_info= "<outputs_network>grid_info.yaml",
    benchmark:
        f"<outputs_transitions>sqra/timing_decomposition.txt"
    output:
        eigenvalues = f"<outputs_transitions>sqra/eigenvalues.npy",
        eigenvectors = f"<outputs_transitions>sqra/eigenvectors.npy"
    run:
        from molgri.molecules.transitions import DecompositionTool

        grid_info = read_object(input.grid_info)
        total_length = int(grid_info["N_total"])
        kept_indices = read_object(input.indices_to_keep)
        my_matrix = read_object(input.reduced_msm)
        print("Decomposing matrix of shape ", my_matrix.shape)

        # calculation
        dt = DecompositionTool(my_matrix, kept_indices, total_length)
        all_eigenval, all_eigenvec = dt.decompose_sqra()

        write_object(all_eigenval, output.eigenvalues)
        write_object(all_eigenvec,output.eigenvectors)

rule plot_sqra_eigenvectors_as_lines:
    input:
        eigenvectors = f"<outputs_transitions>sqra/eigenvectors.npy",
    output:
        plot = f"<outputs_other_plots>eigenvectors_sqra.png"
    run:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        eigenvector_array = read_object(input.eigenvectors)

        N_interesting_eigenvectors = config["msm"]["num_interesting_eigenvectors"]

        fig = make_subplots(rows=N_interesting_eigenvectors,cols=1)

        for row in range(N_interesting_eigenvectors):
            fig.add_trace(
                go.Scatter(x=np.arange(eigenvector_array.shape[0]),y=eigenvector_array[:, row], line=dict(color="black"),
                    mode="lines"),row=1+row,col=1)

        fig.write_image(output.plot, scale=3)

# TODO: display (reduced) rate matrix and display the stucture

rule display_rate_matrix:
    input:
        rate_matrix = f"<outputs_transitions>sqra/sqra.npz",
        reduced_sqra= f"<outputs_transitions>sqra/reduced_sqra.npz",
        indices_to_keep= f"<outputs_transitions>sqra/indices_to_keep.npy",
    output:
        plot = f"<outputs_other_plots>array_sqra.png",
        plot_reduced = f"<outputs_other_plots>array_reduced_sqra.png",
    run:
        show_array(read_object(input.rate_matrix).toarray(), "Rate Matrix",
            save_as=output.plot, show=False)
        show_array(read_object(input.reduced_sqra).toarray(),"Reduced Rate Matrix",
            save_as=output.plot_reduced,show=False, indices=read_object(input.indices_to_keep))

rule plot_eigenvalues:
    input:
        eigenvalues = f"<outputs_transitions>sqra/eigenvalues.npy",
    output:
        plot = f"<outputs_other_plots>eigenvalues_sqra.png"
    run:
        eigenvals = read_object(input.eigenvalues)
        xs = np.linspace(0, 1, num=len(eigenvals))

        fig = go.Figure()

        # vertical lines
        for i, eigenw in enumerate(eigenvals):
            fig.add_shape(type="line", x0=xs[i], y0=0, x1=xs[i], y1=eigenw, line=dict(color="black", width=5),
                          opacity=1)

        # horizontal infinite line
        fig.add_hline(y=0, line=dict(color="black", width=5), opacity=1)

        # plotting (after the axes so that the numbers are on top if there is any overlap)
        fig.add_scatter(x=xs, y=eigenvals, mode='markers+text', text=[f"{el:.3e}" for el in eigenvals],
                        marker=dict(size=14, color="black"), opacity=1)

        #self.fig.update_yaxes(title="Eigenvalues")
        fig.update_layout(xaxis_visible=False, xaxis_showticklabels=False, font=dict(size=20), yaxis_visible=False)

        fig.update_traces(textposition='bottom center', textfont=dict(size=24))

        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),  # Reduce margins (left, right, top, bottom)
            title_x=0.5,  # Center title
            autosize=True,  # Allow automatic resizing
            xaxis=dict(automargin=True),  # Auto-adjust x-axis margins
            yaxis=dict(automargin=True)  # Auto-adjust y-axis margins
        )
        fig.write_image(output.plot, scale=3)
