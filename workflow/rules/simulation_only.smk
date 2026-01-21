import numpy as np
import pandas as pd
from numpy._typing import NDArray

rule gromacs_equilibration:
    """
    This rule gets structure, trajectory, topology and gromacs run file as input, as output we are only interested in
    energies.
    """
    input:
        structure=f"<outputs_gromacs>structure.<ext_str>",
        runfile_minim=f"<outputs_gromacs>minim.mdp",
        runfile_nvt=f"<outputs_gromacs>nvt.mdp",
        index=f"<outputs_gromacs>index.ndx",
        topology=f"<outputs_gromacs>topol.top",
        force_field_stuff=f"<outputs_gromacs>force_field_stuff/"
    shadow: "minimal"
    output:
        energy=f"<outputs_gromacs>nvt.gro",
    shell:
        """
        initial_dir=$(pwd)
        cd $(dirname {input.runfile_minim})
        echo $(pwd)
        export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
        gmx22 grompp -f $(basename {input.runfile_minim}) -c $(basename {input.structure}) -p $(basename {input.topology}) -o em.tpr -r $(basename {input.structure})
        gmx22 mdrun -v -deffnm em

        gmx22 grompp -f $(basename {input.runfile_nvt}) -c em.gro -r em.gro -p $(basename {input.topology}) -o nvt.tpr -n $(basename {input.index})
        gmx22 mdrun -v -deffnm nvt

        cd "$initial_dir" || exit
        """


rule gromacs_production:
    """
    This rule gets structure, trajectory, topology and gromacs run file as input, as output we are only interested in
    energies.
    """
    input:
        structure=f"<outputs_gromacs>nvt.<ext_str>",
        runfile=f"<outputs_gromacs>production.mdp",
        topology=f"<outputs_gromacs>topol.top",
        select_energy=f"<outputs_gromacs>select_energy",
        force_field_stuff=f"<outputs_gromacs>force_field_stuff/",
        index=f"<outputs_gromacs>index.ndx",
    log:
        log=f"<outputs_gromacs>logging_gromacs.log"
    benchmark:
        repeat(f"<outputs_gromacs>gromacs_benchmark.txt",1)
    shadow: "minimal"
    output:
        structure_tpr=f"<outputs_gromacs>structure.tpr",
        energy=f"<outputs_gromacs>energy.xvg",
        original_trajectory=f"<outputs_gromacs>raw_trajectory.<ext_trj>"
    shell:
        """
        initial_dir=$(pwd)
        cd $(dirname {input.runfile})
        export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
        gmx22 grompp -f $(basename {input.runfile}) -c $(basename {input.structure}) -r $(basename {input.structure}) -p $(basename {input.topology}) -o $(basename {output.structure_tpr}) -n $(basename {input.index})
        gmx22 mdrun -v -deffnm structure -g $(basename {log.log})
        mv structure.xtc $(basename {output.original_trajectory})
        gmx22 energy -f structure.edr -o $(basename {output.energy}) < $(basename {input.select_energy})
        cd "$initial_dir" || exit
        """


rule postprocess_gromacs:
    input:
        original_trajectory = f"<outputs_gromacs>raw_trajectory.<ext_trj>",
        structure_tpr=f"<outputs_gromacs>structure.tpr",
        index=f"<outputs_gromacs>index.ndx",
    log:
        log=f"<outputs_gromacs>logging_gromacs.log"
    benchmark:
        repeat(f"<outputs_gromacs>gromacs_benchmark.txt",1)
    shadow: "minimal"
    output:
        trajectory=f"<outputs_gromacs>trajectory.<ext_trj>",
    shell:
        """
        initial_dir=$(pwd)
        cd $(dirname {input.original_trajectory})
        export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
        # now fit to first frame
        echo "2\n0\n" |  gmx22 trjconv -f $(basename {input.original_trajectory}) -s  $(basename {input.structure_tpr}) -pbc mol -center -o centered_trajectory.xtc -n $(basename {input.index})
        echo "2\n0\n" |  gmx22 trjconv -fit rot+trans -f centered_trajectory.xtc -o $(basename {output.trajectory}) -s  $(basename {input.structure_tpr}) -n $(basename {input.index})
        cd "$initial_dir" || exit
        """


rule wrap_molecule2_COM:
    input:
        structure = f"<outputs_gromacs>structure.<ext_str>",
        trajectory = f"<outputs_gromacs>trajectory.<ext_trj>",
        structure1 = f"<outputs_gromacs>molecule1.<ext_str>",
    output:
        com_m2 = f"<outputs_assignment>m2_com.npy",
        com_m2_wrapped = f"<outputs_assignment>cuboid_wrapped_m2_com.npy",
    run:
        from workflow.helpers.io import get_atomgoup_m1, get_atomgoup_m2, write_object
        from molgri.molecules.find_unit_cell import get_rectangular_cell_side_lengths, wrap_to_cuboid_cell
        from MDAnalysis import Universe

        # determine com of 2nd molecule
        u = Universe(input.structure, input.trajectory)
        ag_m2 = get_atomgoup_m2(u, input.structure1)
        ag_m1 = get_atomgoup_m1(u, input.structure1)

        com_array_m1 = np.zeros((len(u.trajectory), 3))
        com_array_m2 = np.zeros((len(u.trajectory), 3))
        for i, ts in enumerate(u.trajectory):
            shift = ag_m1.center_of_mass()
            u.atoms.translate(-shift)
            com_array_m1[i] = ag_m1.center_of_mass()
            com_array_m2[i] = ag_m2.center_of_mass()

        write_object(com_array_m2, output.com_m2)
        # assert com of m1 not changing
        assert np.max(com_array_m1 - com_array_m1[0]) < 0.01, "Molecule 1 seems to be moving - is it not fitted to the reference or just very flexible?"


        # determine cuboid cell
        side_lengths = get_rectangular_cell_side_lengths(input.structure1)
        origin = np.zeros(3)

        # wrap
        wrapped_com_m2 = wrap_to_cuboid_cell(origin, side_lengths, com_array_m2)
        write_object(wrapped_com_m2, output.com_m2_wrapped)

