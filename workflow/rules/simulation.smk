import numpy as np
import pandas as pd
from numpy.typing import NDArray
from MDAnalysis import Universe, Writer

from molgri.molecules.find_unit_cell import get_rectangular_cell_side_lengths, wrap_multiple_atoms_to_cuboid_cell
from workflow.helpers.io import read_object, write_object, get_num_atoms, from_xvg_to_csv_energy

# rule gromacs_equilibration:
#     """
#     This rule gets structure, trajectory, topology and gromacs run file as input, as output we are only interested in
#     energies.
#     """
#     input:
#         structure=f"<outputs_gromacs>structure.<ext_str>",
#         runfile_minim=f"<outputs_gromacs>minim.mdp",
#         runfile_nvt=f"<outputs_gromacs>nvt.mdp",
#         index=f"<outputs_gromacs>index.ndx",
#         topology=f"<outputs_gromacs>topol.top",
#         force_field_stuff=f"<outputs_gromacs>force_field_stuff/"
#     shadow: "minimal"
#     output:
#         energy=f"<outputs_gromacs>nvt.gro",
#     shell:
#         """
#         initial_dir=$(pwd)
#         cd $(dirname {input.runfile_minim})
#         echo $(pwd)
#         export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
#         gmx22 grompp -f $(basename {input.runfile_minim}) -c $(basename {input.structure}) -p $(basename {input.topology}) -o em.tpr -r $(basename {input.structure})
#         gmx22 mdrun -v -deffnm em
#
#         gmx22 grompp -f $(basename {input.runfile_nvt}) -c em.gro -r em.gro -p $(basename {input.topology}) -o nvt.tpr -n $(basename {input.index})
#         gmx22 mdrun -v -deffnm nvt
#
#         cd "$initial_dir" || exit
#         """
#
#
# rule gromacs_production:
#     """
#     This rule gets structure, trajectory, topology and gromacs run file as input, as output we are only interested in
#     energies.
#     """
#     input:
#         structure=f"<outputs_gromacs>nvt.<ext_str>",
#         runfile=f"<outputs_gromacs>production.mdp",
#         topology=f"<outputs_gromacs>topol.top",
#         select_energy=f"<outputs_gromacs>select_energy",
#         force_field_stuff=f"<outputs_gromacs>force_field_stuff/",
#         index=f"<outputs_gromacs>index.ndx",
#     log:
#         log=f"<outputs_gromacs>logging_gromacs.log"
#     benchmark:
#         repeat(f"<outputs_gromacs>gromacs_benchmark.txt",1)
#     shadow: "minimal"
#     output:
#         structure_tpr=f"<outputs_gromacs>structure.tpr",
#         energy=f"<outputs_gromacs>energy.xvg",
#         original_trajectory=f"<outputs_gromacs>raw_trajectory.<ext_trj>"
#     shell:
#         """
#         initial_dir=$(pwd)
#         cd $(dirname {input.runfile})
#         export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
#         gmx22 grompp -f $(basename {input.runfile}) -c $(basename {input.structure}) -r $(basename {input.structure}) -p $(basename {input.topology}) -o $(basename {output.structure_tpr}) -n $(basename {input.index})
#         gmx22 mdrun -v -deffnm structure -g $(basename {log.log})
#         mv structure.xtc $(basename {output.original_trajectory})
#         gmx22 energy -f structure.edr -o $(basename {output.energy}) < $(basename {input.select_energy})
#         cd "$initial_dir" || exit
#         """


rule postprocess_gromacs:
    input:
        original_trajectory = f"<simulation>raw_trajectory.<ext_trj>",
        structure_tpr=f"<simulation>structure.tpr",
        structure_gro=f"<simulation>structure.gro",
        index=f"<simulation>index.ndx",
    benchmark:
        repeat(f"<simulation>duration_gromacs_postprocessing.txt",1)
    shadow: "minimal"
    output:
        centered_trajectory = f"<simulation>centered_trajectory.<ext_trj>",
        trajectory=f"<simulation>trajectory.<ext_trj>",
    shell:
        """
        initial_dir=$(pwd)
        cd $(dirname {input.original_trajectory})
        export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
        # now fit to first frame
        echo "0\n" |  gmx22 trjconv -f $(basename {input.original_trajectory}) -s  $(basename {input.structure_tpr}) -pbc mol -boxcenter tric -o $(basename {output.centered_trajectory}) -n $(basename {input.index})
        echo "4\n0\n" |  gmx22 trjconv -fit trans -f $(basename {output.centered_trajectory}) -o $(basename {output.trajectory}) -s  $(basename {input.structure_gro}) -n $(basename {input.index})
        cd "$initial_dir" || exit
        """
    #-box 2.467006637598667 4.272980838930551 4.0

rule shortened_trajectory_gromacs:
    """
    This is helpful for testing new analysis methods without waiting forever for results. Just use shortened_trajectory
    instead of trajectory in input.
    """
    input:
        trajectory=f"<simulation>trajectory.<ext_trj>",
        structure_tpr=f"<simulation>structure.tpr",
        index=f"<simulation>index.ndx",
    benchmark:
        repeat(f"<simulation>duration_gromacs_shortening.txt",1)
    shadow: "minimal"
    params:
        length_shortened_trajectory_ps = int(1000 * float(config["analysis"]["length_shortened_trajectory_ns"]))
    output:
        trajectory=f"<simulation>shortened_trajectory.<ext_trj>",
    shell:
        """
        initial_dir=$(pwd)
        cd $(dirname {input.trajectory})
        export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
        echo "0\n" |  gmx22 trjconv -f $(basename {input.trajectory}) -s  $(basename {input.structure_tpr}) -o $(basename {output.trajectory}) -n $(basename {input.index}) -e {params.length_shortened_trajectory_ps}
 cd "$initial_dir" || exit
        """

checkpoint create_energy_csv_trajectory:
    """
    For the pseudotrajectory, read the energy of each frame.
    """
    input:
        energy="<simulation>energy.xvg",
    output:
        energy_csv = "<simulation>energy.csv"
    run:
        from_xvg_to_csv_energy(input.energy, output.energy_csv)

