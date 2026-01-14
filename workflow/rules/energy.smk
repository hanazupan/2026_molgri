from workflow.helpers.io import read_object, write_object


checkpoint create_energy_csv:
    input:
        energy=f"<outputs_gromacs>energy.xvg",
    output:
        energy_csv = f"<outputs>energy.csv"
    run:
        import pandas as pd
        import numpy as np

        my_energy = read_object(input.energy)
        coulumb = my_energy["Coul-SR:MOL1-MOL2"].to_numpy()
        lj = my_energy["LJ-SR:MOL1-MOL2"].to_numpy()


        df = pd.DataFrame(np.array([coulumb, lj]).T,
            columns=["Coulomb contribution [kJ/mol]", "Lennard-Jones contribution [kJ/mol]"])

        df["Binding energy [kJ/mol]"] = df["Coulomb contribution [kJ/mol]"] + df["Lennard-Jones contribution [kJ/mol]"]
        df.index.name = "Total index"
        write_object(df, output.energy_csv)


rule print_lowest_energies:
    """
    Use this rule if you want to quickly look at the indices of the lowest energies.
    """
    input:
        energy_csv =f"<outputs>energy.csv"
    run:
        df = read_object(input.energy_csv)
        df = df.sort_values(by="Binding energy [kJ/mol]",ascending=True)
        print(df.head(50))

rule lowest_E_indices:
    """
    Write in a .txt file where the N lowest energy indices are written down (eg for later plotting).
    """
    input:
        energy_csv = f"<outputs>energy.csv"
    output:
        indices= f"<outputs_indices>lowest_{{N}}_binding_energies.txt"
    run:
        df_energy = read_object(input.energy_csv)
        required_indices = np.array(df_energy.nsmallest(int(wildcards.N), "Binding energy [kJ/mol]").index)
        write_object(required_indices, output.indices)


rule violin_plot_E_distributions:
    input:
        energy = f"<outputs>energy.csv"
    output:
        violin_plot = f"<outputs_other_plots>violin_plot_energies.png"
    run:
        from molgri.plotting import show_violin
        df = read_object(input.energy)

        max_energy = config['plotting']['upper_E_limit']
        energies = df["Binding energy [kJ/mol]"]

        show_violin(energies, max_energy, "Binding Energy", save_as=output.violin_plot, show=False)
