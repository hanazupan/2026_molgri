"""Here are parts of a sqra pipeline that are GROMACS-specific"""


from workflow.helpers.PATHS import PATH_INPUT_DEFAULT_GROMACS, NAME_ENERGY_FOLDER, NAME_PT_FOLDER, NAME_SIMULATION_FOLDER

MOLECULE_NAMES = f"{config['pseudotrajectory']['molecule_1']}_{config['pseudotrajectory']['molecule_2']}/"
PROVIDED_DATA_PATH =  f"{PATH_INPUT_DEFAULT_GROMACS}{MOLECULE_NAMES}"
STRUCTURE_ENDING = config["pseudotrajectory"]["structure_ending"]
TRAJECTORY_ENDING = config["pseudotrajectory"]["trajectory_ending"]

rule all_gromacs:
    input:
        f"{{some_path}}energy.xvg"

rule copy_gromacs_input:
    """
    Copy the rest of necessary files to start a sqra run with a GROMACS calculation and potentially adapt default mdrun params.
    """
    wildcard_constraints:
        output_folder=fr"({NAME_SIMULATION_FOLDER}|{NAME_ENERGY_FOLDER})"
    input:
        dimer_topology = f"{PROVIDED_DATA_PATH}topol.top",
        select_energy = f"{PROVIDED_DATA_PATH}select_energy_five",
        runfile= f"{PROVIDED_DATA_PATH}mdrun.mdp",
        force_field_stuff = f"{PROVIDED_DATA_PATH}force_field_stuff/"
    output:
        dimer_topology = f"{{some_path}}{{output_folder}}topol.top",
        select_energy = f"{{some_path}}{{output_folder}}select_energy_five",
        runfile = f"{{some_path}}{{output_folder}}mdrun.mdp",
        force_field_stuff = directory(f"{{some_path}}{{output_folder}}force_field_stuff/")
    run:
        import shutil
        shutil.copy(input.select_energy,output.select_energy)
        shutil.copy(input.dimer_topology, output.dimer_topology)
        shutil.copy(input.select_energy,output.select_energy)
        shutil.copy(input.runfile,output.runfile)
        shutil.copytree(input.force_field_stuff,output.force_field_stuff, dirs_exist_ok=True)

rule copy_additional_simulation_files:
    """
    Copy the rest of necessary files to start a sqra run with a GROMACS calculation and potentially adapt default mdrun params.
    """
    wildcard_constraints:
        output_folder=fr"({NAME_SIMULATION_FOLDER}|{NAME_ENERGY_FOLDER})"
    input:
        minim_runfile=f"{PROVIDED_DATA_PATH}minim.mdp",
        runfile_nvt= f"{PROVIDED_DATA_PATH}nvt.mdp",
        index= f"{PROVIDED_DATA_PATH}index.ndx",
    output:
        minim_runfile=f"{{some_path}}{{output_folder}}minim.mdp",
        runfile_nvt= f"{{some_path}}{{output_folder}}nvt.mdp",
        index= f"{{some_path}}{{output_folder}}index.ndx",
    run:
        import shutil
        shutil.copy(input.minim_runfile,output.minim_runfile)
        shutil.copy(input.runfile_nvt,output.runfile_nvt)
        shutil.copy(input.index,output.index)


rule copy_gromacs_monomer_input:
    """
    Copy the rest of necessary files to start a sqra run with a GROMACS calculation and potentially adapt default mdrun params.
    """
    input:
        m1_topology = f"{PROVIDED_DATA_PATH}topol_m{{i}}.top",
        select_energy = f"{PROVIDED_DATA_PATH}select_energy_five",
        runfile= f"{PROVIDED_DATA_PATH}mdrun.mdp",
        force_field_stuff = f"{PROVIDED_DATA_PATH}force_field_stuff/"
    output:
        m1_topology= f"{{some_path}}molecule{{i}}/topol.top",
        select_energy = f"{{some_path}}molecule{{i}}/select_energy_five",
        runfile = f"{{some_path}}molecule{{i}}/mdrun.mdp",
        force_field_stuff = directory(f"{{some_path}}molecule{{i}}/force_field_stuff/")
    run:
        import shutil
        shutil.copy(input.select_energy,output.select_energy)
        shutil.copy(input.m1_topology,output.m1_topology)
        shutil.copy(input.runfile,output.runfile)
        shutil.copytree(input.force_field_stuff,output.force_field_stuff, dirs_exist_ok=True)

rule gromacs_monomer_energy:
    wildcard_constraints:
        ENERGY_PROGRAM  = "GROMACS"
    input:
        structure = f"{{some_path}}{NAME_PT_FOLDER}molecule{{i}}.{STRUCTURE_ENDING}",
        runfile = f"{{some_path}}molecule{{i}}/mdrun.mdp",
        topology = f"{{some_path}}molecule{{i}}/topol.top",
        select_energy = f"{{some_path}}molecule{{i}}/select_energy_five",
        force_field_stuff = f"{{some_path}}molecule{{i}}/force_field_stuff/"
    #shadow: "minimal"
    log:
        log = f"{{some_path}}molecule{{i}}/logging_gromacs.log"
    benchmark:
        repeat(f"{{some_path}}molecule{{i}}/gromacs_benchmark.txt", 1)
    output:
        energy = f"{{some_path}}molecule{{i}}/energy.xvg",
    shell:
        """
        initial_dir=$(pwd)
        cd $(dirname {input.runfile})
        export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
        gmx22 grompp -f $(basename {input.runfile}) -c ../pseudotrajectory/$(basename {input.structure}) -p $(basename {input.topology}) -o result.tpr  -maxwarn 1
        gmx22 mdrun -s result.tpr -rerun ../pseudotrajectory/$(basename {input.structure}) -g $(basename {log.log})
        gmx22 energy -f ener.edr -o $(basename {output.energy}) < $(basename {input.select_energy})
        cd "$initial_dir" || exit
        """

rule gromacs_rerun:
    """
    This rule gets structure, trajectory, topology and gromacs run file as input, as output we are only interested in
    energies.
    """
    input:
        structure = f"{{some_path}}{NAME_PT_FOLDER}structure.{STRUCTURE_ENDING}",
        trajectory = f"{{some_path}}{NAME_PT_FOLDER}trajectory.{TRAJECTORY_ENDING}",
        runfile = f"{{some_path}}{NAME_ENERGY_FOLDER}mdrun.mdp",
        topology = f"{{some_path}}{NAME_ENERGY_FOLDER}topol.top",
        select_energy = f"{{some_path}}{NAME_ENERGY_FOLDER}select_energy_five",
        force_field_stuff = f"{{some_path}}{NAME_ENERGY_FOLDER}force_field_stuff/"
    shadow: "minimal"
    log:
        log = f"{{some_path}}{NAME_ENERGY_FOLDER}logging_gromacs.log"
    benchmark:
        repeat(f"{{some_path}}{NAME_ENERGY_FOLDER}gromacs_benchmark.txt", 1)
    output:
        energy = f"{{some_path}}{NAME_ENERGY_FOLDER}energy.xvg",
    shell:
        """
        initial_dir=$(pwd)
        cd $(dirname {input.runfile})
        export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
        gmx22 grompp -f $(basename {input.runfile}) -c ../pseudotrajectory/$(basename {input.structure}) -p $(basename {input.topology}) -o result.tpr
        gmx22 mdrun -s result.tpr -rerun ../pseudotrajectory/$(basename {input.trajectory}) -g $(basename {log.log})
        gmx22 energy -f ener.edr -o $(basename {output.energy}) < $(basename {input.select_energy})
        cd "$initial_dir" || exit
        """

rule gromacs_equilibration:
    """
    This rule gets structure, trajectory, topology and gromacs run file as input, as output we are only interested in
    energies.
    """
    input:
        structure = f"{{some_path}}{NAME_SIMULATION_FOLDER}structure.{STRUCTURE_ENDING}",
        runfile_minim = f"{{some_path}}{NAME_SIMULATION_FOLDER}minim.mdp",
        runfile_nvt = f"{{some_path}}{NAME_SIMULATION_FOLDER}nvt.mdp",
        index = f"{{some_path}}{NAME_SIMULATION_FOLDER}index.ndx",
        topology = f"{{some_path}}{NAME_SIMULATION_FOLDER}topol.top",
        select_energy = f"{{some_path}}{NAME_SIMULATION_FOLDER}select_energy_five",
        force_field_stuff = f"{{some_path}}{NAME_SIMULATION_FOLDER}force_field_stuff/"
    shadow: "minimal"
    output:
        energy = f"{{some_path}}{NAME_SIMULATION_FOLDER}nvt.gro",
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
        structure=f"{{some_path}}{NAME_SIMULATION_FOLDER}nvt.{STRUCTURE_ENDING}",
        runfile=f"{{some_path}}{NAME_SIMULATION_FOLDER}mdrun.mdp",
        topology=f"{{some_path}}{NAME_SIMULATION_FOLDER}topol.top",
        select_energy=f"{{some_path}}{NAME_SIMULATION_FOLDER}select_energy_five",
        force_field_stuff=f"{{some_path}}{NAME_SIMULATION_FOLDER}force_field_stuff/"
    shadow: "minimal"
    log:
        log=f"{{some_path}}{NAME_SIMULATION_FOLDER}logging_gromacs.log"
    benchmark:
        repeat(f"{{some_path}}{NAME_SIMULATION_FOLDER}gromacs_benchmark.txt",1)
    output:
        energy=f"{{some_path}}{NAME_SIMULATION_FOLDER}energy.xvg",
    shell:
        """
        initial_dir=$(pwd)
        cd $(dirname {input.runfile})
        export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
        gmx22 grompp -f $(basename {input.runfile}) -c $(basename {input.structure}) -p $(basename {input.topology}) -o result.tpr
        gmx22 mdrun -s result.tpr -g $(basename {log.log})
        gmx22 energy -f ener.edr -o $(basename {output.energy}) < $(basename {input.select_energy})
        cd "$initial_dir" || exit
        """