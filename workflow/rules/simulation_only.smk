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
        original_trajectory=f"<outputs_gromacs>raw_trajectory.xtc",
        trajectory=f"<outputs_gromacs>trajectory.xtc",
    shell:
        """
        initial_dir=$(pwd)
        cd $(dirname {input.runfile})
        export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
        gmx22 grompp -f $(basename {input.runfile}) -c $(basename {input.structure}) -p $(basename {input.topology}) -o raw_trajectory.tpr -n $(basename {input.index})
        gmx22 mdrun -v -deffnm raw_trajectory -g $(basename {log.log})
        gmx22 energy -f raw_trajectory.edr -o $(basename {output.energy}) < $(basename {input.select_energy})
        # now fit to first frame
        gmx22 grompp -f $(basename {input.runfile}) -c $(basename {input.structure}) -p $(basename {input.topology}) -o $(basename {output.structure_tpr}) -n $(basename {input.index}) 
        echo "2\n0\n" |  gmx22 trjconv -f raw_trajectory.xtc -s  $(basename {output.structure_tpr}) -pbc mol -center -o centered_trajectory.xtc -n $(basename {input.index})
        echo "2\n0\n" |  gmx22 trjconv -fit rot+trans -f centered_trajectory.xtc -o $(basename {output.trajectory}) -s  $(basename {output.structure_tpr}) -n $(basename {input.index})
        cd "$initial_dir" || exit
        """