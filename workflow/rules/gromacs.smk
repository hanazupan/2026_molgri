"""Here are parts of a sqra pipeline that are GROMACS-specific"""


from workflow.helpers.PATHS import PATH_INPUT_DEFAULT_GROMACS, NAME_ENERGY_FOLDER

MOLECULE_NAMES = f"{config['pseudotrajectory']['molecule_1']}_{config['pseudotrajectory']['molecule_2']}"
PROVIDED_DATA_PATH =  f"{PATH_INPUT_DEFAULT_GROMACS}{MOLECULE_NAMES}/"

rule all_gromacs:
    input:
        f"{{some_path}}energy.xvg"

rule copy_gromacs_input:
    """
    Copy the rest of necessary files to start a sqra run with a GROMACS calculation and potentially adapt default mdrun params.
    """
    input:
        dimer_topology = f"{PROVIDED_DATA_PATH}topol.top",
        select_energy = f"{PROVIDED_DATA_PATH}select_energy_five",
        runfile= f"{PROVIDED_DATA_PATH}mdrun.mdp",
        force_field_stuff = f"{PROVIDED_DATA_PATH}force_field_stuff/"
    output:
        dimer_topology = f"{{some_path}}{NAME_ENERGY_FOLDER}topol.top",
        select_energy = f"{{some_path}}{NAME_ENERGY_FOLDER}select_energy",
        runfile = f"{{some_path}}{NAME_ENERGY_FOLDER}mdrun.mdp",
        force_field_stuff = directory(f"{{some_path}}{NAME_ENERGY_FOLDER}force_field_stuff/")
    run:
        import shutil
        shutil.copy(input.select_energy,output.select_energy)
        shutil.copy(input.dimer_topology, output.dimer_topology)
        shutil.copy(input.runfile,output.runfile)
        shutil.copytree(input.force_field_stuff,output.force_field_stuff, dirs_exist_ok=True)

rule gromacs_rerun:
    """
    This rule gets structure, trajectory, topology and gromacs run file as input, as output we are only interested in
    energies.
    """
    wildcard_constraints:
        ENERGY_PROGRAM  = "GROMACS"
    input:
        structure = rules.create_pseudotrajectory.output.structure,
        trajectory = rules.create_pseudotrajectory.output.trajectory,
        runfile = rules.copy_gromacs_input.output.runfile,
        topology = rules.copy_gromacs_input.output.dimer_topology,
        select_energy = rules.copy_gromacs_input.output.select_energy,
        force_field_stuff = rules.copy_gromacs_input.output.force_field_stuff
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
