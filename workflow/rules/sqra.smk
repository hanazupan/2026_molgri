import numpy as np
import pandas as pd
from networkx.algorithms.threshold import eigenvectors

from molgri.images.plotting import show_array
from molgri.molecules.rate_merger import delete_rows_columns, expand_eigenvector_to_full_length
from molgri.molecules.transitions import SQRA
from molgri.utils.arrays import k_argmax_in_array, k_argmin_in_array
from workflow.helpers.io import read_object, write_object
from molgri.molecules.transitions import DecompositionTool

rule make_sqra:
    input:
        energies = "<pseudosimulation>energy.csv",
        volumes = "<outputs_network>volumes.npy",
        distances= "<outputs_network>distances.npz",
        surfaces= "<outputs_network>surfaces.npz"
    output:
        rate_matrix = f"<outputs_transitions>sqra/sqra.npz",
        # reduced_sqra= f"<outputs_transitions>sqra/reduced_sqra.npz",
        # indices_to_keep= f"<outputs_transitions>sqra/indices_to_keep.npy",
    params:
        T_in_K = 293,
        diffusion_coefficient = 1,
        capping_factor = config["msm"]["capping_factor"],
        cutting_factor= config["msm"]["cutting_factor"]
    run:
        my_energy = read_object(input.energies)
        my_energy_array = my_energy["Energy [kJ/mol]"].to_numpy()
        volumes = read_object(input.volumes)
        distances = read_object(input.distances)
        surfaces = read_object(input.surfaces)

        # get rate matrix first, then try different cutting factors
        sqra = SQRA(energies=my_energy_array,volumes=volumes,distances=distances,surfaces=surfaces)
        rate_matrix = sqra.get_rate_matrix(params.diffusion_coefficient,params.T_in_K,
            capping_factor="None")
        write_object(rate_matrix, output.rate_matrix)

        # for cutting_factor in range(5000, 4000, -200):
        #     reduced_rate_matrix, to_keep = sqra.cut_high_energy_states(cutting_factor=cutting_factor)
        #     print(cutting_factor, np.max(np.sum(reduced_rate_matrix, axis=1)))
        #     # we demand the row-sum of rate matrix to be close to zero in order to get eigenvectors that behave as such
        #     if np.abs(np.max(np.sum(reduced_rate_matrix, axis=1))) < 1e-8:
        #         print("FOUND CUTTING FACTOR ", cutting_factor)
        #         break
            # print(cutting_factor, )
            #
            # dt = DecompositionTool(rate_matrix,to_keep,len(volumes))
            # try:
            #     print("trying")
            #     # WHY IS THE SUM OF FIRST EIGENVECTOR NOT 1??
            #     all_eigenval, all_eigenvec = dt.decompose_sqra(k=2, maxiter=100)
            #     all_eigenvec_t = all_eigenvec.T
            #     first_sum_is_one = np.isclose(np.sum(all_eigenvec_t[0, :]), 1, atol=1e-3, rtol=1e-3) or  np.isclose(np.sum(all_eigenvec[0, :]), -1, atol=1e-3, rtol=1e-3)
            #
            #     #other_sum_is_zero = np.allclose([np.sum(other_rows) for other_rows in all_eigenvec_t[1:]], 0, atol=1e-3, rtol=1e-3)
            #     print(np.sum(all_eigenvec_t[0, :]))
            #     if first_sum_is_one: #and other_sum_is_zero:
            #         print("FOUND CUTTING FACTOR ", cutting_factor)
            #         break
            # except:
            #     pass
        # else:
        #     raise ValueError("No cutting factor works")


        # print("Reduced Rate Matrix ", len(volumes), reduced_rate_matrix.shape, len(to_keep))
        # write_object(reduced_rate_matrix,output.reduced_sqra)
        # write_object(np.array(to_keep),output.indices_to_keep)



rule reduce_sqra_size:
    input:
        sqra= f"<outputs_transitions>sqra/sqra.npz",
    output:
        reduced_sqra= f"<outputs_transitions>sqra/reduced_sqra.npz",
        indices_to_keep = f"<outputs_transitions>sqra/indices_to_keep.npy",
    params:
        cutting_factor = config["msm"]["cutting_factor"]
    run:
        sqra = read_object(input.sqra)

        logarithmically_spaced_cutting_factors = np.logspace(2, 300, num=200)
        logarithmically_spaced_cutting_factors = logarithmically_spaced_cutting_factors [::-1]

        for cutting_factor in logarithmically_spaced_cutting_factors:
            reduced_sqra, indices_to_keep = delete_rows_columns(sqra,"sqra", cutting_factor)
            print(cutting_factor, np.max(np.sum(reduced_sqra, axis=1)))
            # we demand the row-sum of rate matrix to be close to zero in order to get eigenvectors that behave as such
            if np.abs(np.max(np.sum(reduced_sqra, axis=1))) < 1e-12:
                print("FOUND CUTTING FACTOR ", cutting_factor)
                break


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
    params:
        tolerance = config["msm"]["tolerance"],
        sigma_sqra = config["msm"]["sigma_sqra"]
    run:
        grid_info = read_object(input.grid_info)
        total_length = int(grid_info["N_total"])
        kept_indices = read_object(input.indices_to_keep)
        my_matrix = read_object(input.reduced_msm)
        # print("Decomposing matrix of shape ", my_matrix.shape)
        # from scipy.linalg import eig
        # print("NORMAL DECOMPOSITION")
        # eigenval, left_eigenvec, right_eigenvec = eig(my_matrix.toarray(), left=True, right=True)
        # print("Imaginary ", eigenval.imag.max(), left_eigenvec.imag.max(), right_eigenvec.imag.max())
        # eigenval = eigenval.real
        # idx = eigenval.argsort()[::-1]
        # eigenval = eigenval[idx]
        # # typically we want left eigenvectors for sqra
        # #print("Eigenvalues: ",eigenval)
        # left_eigenvec = left_eigenvec.real
        # left_eigenvec = left_eigenvec[:, idx]
        #
        #
        # expanded_eigenvectors = []
        # # expand to full length
        # allowed_indices = [0]
        # for i, eigenvector in enumerate(left_eigenvec.T):
        #     expanded_eigenvector = expand_eigenvector_to_full_length(eigenvector, kept_indices, total_length)
        #     if i==0:
        #         expanded_eigenvectors.append(expanded_eigenvector)
        #         # should we demand they sum up to zero?
        #         print(i, np.sum(expanded_eigenvector))
        #     if np.isclose(np.sum(expanded_eigenvector), 0, atol=1e-2, rtol=1e-2):
        #         allowed_indices.append(i)
        #         expanded_eigenvectors.append(expanded_eigenvector)
        #         # should we demand they sum up to zero?
        #         print(i, np.sum(expanded_eigenvector),  k_argmax_in_array(expanded_eigenvector, 7), k_argmin_in_array(expanded_eigenvector, 7))
        # expanded_eigenvectors = np.array(expanded_eigenvectors).T
        # print("shape ", expanded_eigenvectors.shape)
        # calculation
        dt = DecompositionTool(my_matrix, kept_indices, total_length)
        print(my_matrix.shape)
        all_eigenval, all_eigenvec = dt.decompose_sqra(sigma=float(params.sigma_sqra), tolerance=float(params.tolerance))

        write_object(all_eigenval, output.eigenvalues)
        write_object(all_eigenvec,output.eigenvectors)

        # print("allowed_eigenvalues ", eigenval[allowed_indices])
        #
        # write_object(eigenval[allowed_indices],output.eigenvalues)
        # write_object(expanded_eigenvectors,output.eigenvectors)

rule plot_sqra_eigenvectors_as_lines:
    input:
        eigenvectors = f"<outputs_transitions>sqra/eigenvectors.npy",
    output:
        plot = f"<outputs_other_plots>eigenvectors_sqra.png"
    params:
        N_interesting_eigenvectors = config["msm"]["num_interesting_eigenvectors"]
    run:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        eigenvector_array = read_object(input.eigenvectors)

        N_interesting_eigenvectors = params.N_interesting_eigenvectors

        fig = make_subplots(rows=N_interesting_eigenvectors,cols=1)

        for row in range(N_interesting_eigenvectors):
            fig.add_trace(
                go.Scatter(x=np.arange(eigenvector_array.shape[0]),y=eigenvector_array[:, row], line=dict(color="black"),
                    mode="lines"),row=1+row,col=1)
            # todo names of peaks
        fig.add_hline(
            y=0,
            line_color="red",
            line_width=2,
            line_dash="dash"
        )
        fig.update_layout(showlegend=False,plot_bgcolor="white",)
        fig.write_image(output.plot, scale=3)

rule display_rate_matrix:
    input:
        rate_matrix = f"<outputs_transitions>sqra/sqra.npz",
        reduced_sqra= f"<outputs_transitions>sqra/reduced_sqra.npz",
        indices_to_keep= f"<outputs_transitions>sqra/indices_to_keep.npy",
    output:
        plot = f"<outputs_other_plots>array_sqra.png",
        plot_reduced = f"<outputs_other_plots>array_reduced_sqra.png",
    run:
        np.set_printoptions(precision=7,suppress=True,linewidth=np.inf)
        as_array = read_object(input.rate_matrix).toarray()
        show_array(as_array, "Rate Matrix",
            save_as=output.plot, show=False)
        # print(as_array[7, :],)
        # print(as_array[8, 7],)
        # print(as_array[9, 0], as_array[9, 10],)
        # print(as_array[44, 35], as_array[44, 43], as_array[44, 53],)
        # print(as_array[53, 44], as_array[53, 52], as_array[53, 62],)
        for i, row in enumerate(as_array):
            if i in [7, 8, 9, 43, 44, 53]:
                print(i, np.unique(row))
        print("---- REDUCED ---")
        as_array = read_object(input.reduced_sqra).toarray()
        show_array(as_array,title="Reduced Rate Matrix",
            save_as=output.plot_reduced,show=False, indices=read_object(input.indices_to_keep))
        for i, row in enumerate(as_array):
            if i in [7, 8, 9, 43, 44, 53]:
                print(i, np.unique(row))

rule plot_eigenvalues:
    input:
        eigenvalues = f"<outputs_transitions>sqra/eigenvalues.npy",
    output:
        plot = f"<outputs_other_plots>eigenvalues_sqra.png"
    run:
        np.set_printoptions(precision=16,suppress=True,linewidth=np.inf)
        eigenvals = read_object(input.eigenvalues)
        print(eigenvals)

        max_num = 8

        xs = np.linspace(0, 1, num=max_num)

        fig = go.Figure()

        # vertical lines
        for i, eigenw in enumerate(eigenvals[:max_num]):
            fig.add_shape(type="line", x0=xs[i], y0=0, x1=xs[i], y1=eigenw, line=dict(color="black", width=5),
                          opacity=1)

        # horizontal infinite line
        fig.add_hline(y=0, line=dict(color="black", width=5), opacity=1)

        # plotting (after the axes so that the numbers are on top if there is any overlap)
        fig.add_scatter(x=xs, y=eigenvals[:max_num], mode='markers+text', text=[f"{el:.1e}" for el in eigenvals[:max_num]],
                        marker=dict(size=14, color="black"), opacity=1)

        #self.fig.update_yaxes(title="Eigenvalues")
        fig.update_layout(xaxis_visible=False, xaxis_showticklabels=False, font=dict(size=20), yaxis_visible=False)

        fig.update_traces(textposition='bottom center', textfont=dict(size=24))

        fig.update_layout(
            margin=dict(l=100, r=40, t=40, b=120),  # Reduce margins (left, right, top, bottom)
            #title_x=0.5,  # Center title
            #autosize=True,  # Allow automatic resizing
            #xaxis=dict(automargin=True),  # Auto-adjust x-axis margins
            #yaxis=dict(automargin=True),  # Auto-adjust y-axis margins
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        fig.update_layout(
            width=1000,
            height=700
        )
        fig.write_image(output.plot, scale=3)
