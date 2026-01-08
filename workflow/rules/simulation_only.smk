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
        energy=f"<outputs_gromacs>energy.xvg",
        trajectory=f"<outputs_gromacs>trajectory.xtc",
    shell:
        """
        initial_dir=$(pwd)
        cd $(dirname {input.runfile})
        export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
        gmx22 grompp -f $(basename {input.runfile}) -c $(basename {input.structure}) -p $(basename {input.topology}) -o trajectory.tpr -n $(basename {input.index})
        gmx22 mdrun -v -deffnm trajectory -g $(basename {log.log})
        gmx22 energy -f trajectory.edr -o $(basename {output.energy}) < $(basename {input.select_energy})
        cd "$initial_dir" || exit
        """