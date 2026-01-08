"""Here are parts of a sqra and msm pipeline that are GROMACS-specific"""


from workflow.helpers.PATHS import NAME_GROMACS_FOLDER, NAME_ENERGY_FOLDER, NAME_PT_FOLDER

rule all_gromacs:
    input:
        f"<outputs_gromacs>energy.xvg"

rule copy_mdp_files:
    """
    Copy only mdp files that are required - e.g. for rerun you do not need a minim.mdp and nvt.mdp.
    """
    input:
        mdp = f"<inputs_gromacs>{{file_name}}.mdp",
    output:
        mdp = f"<outputs_gromacs>{{file_name}}.mdp",
    run:
        import shutil
        shutil.copy(input.mdp,output.mdp)


rule copy_other_gromacs_input:
    """
    Copy the rest of necessary files to start a GROMACS calculation.
    """
    input:
        dimer_topology = f"<inputs_gromacs>topol.top",
        select_energy = f"<inputs_gromacs>select_energy",
        index = f"<inputs_gromacs>index.ndx",
        force_field_stuff = f"<inputs_gromacs>force_field_stuff/"
    output:
        dimer_topology = f"<outputs_gromacs>topol.top",
        select_energy = f"<outputs_gromacs>select_energy",
        index = f"<outputs_gromacs>index.ndx",
        force_field_stuff = directory(f"<outputs_gromacs>force_field_stuff/")
    run:
        import shutil
        shutil.copy(input.select_energy,output.select_energy)
        shutil.copy(input.dimer_topology, output.dimer_topology)
        shutil.copy(input.select_energy,output.select_energy)
        shutil.copy(input.index,output.index)
        shutil.copytree(input.force_field_stuff,output.force_field_stuff, dirs_exist_ok=True)

# rule copy_additional_simulation_files:
#     """
#     Copy the rest of necessary files to start a sqra run with a GROMACS calculation and potentially adapt default mdrun params.
#     """
#     wildcard_constraints:
#         output_folder=fr"({NAME_GROMACS_FOLDER}|{NAME_ENERGY_FOLDER}|molecule1/|molecule2/)"
#     input:
#         minim_runfile=f"<inputs_gromacs>minim.mdp",
#         runfile_nvt= f"<inputs_gromacs>nvt.mdp",
#         runfile_production= f"<inputs_gromacs>production.mdp",
#         index= f"<inputs_gromacs>index.ndx",
#     output:
#         minim_runfile=f"<outputs>{{output_folder}}minim.mdp",
#         runfile_nvt= f"<outputs>{{output_folder}}nvt.mdp",
#         runfile_production = f"<outputs>{{output_folder}}production.mdp",
#         index= f"<outputs>{{output_folder}}index.ndx",
#     run:
#         import shutil
#         shutil.copy(input.minim_runfile,output.minim_runfile)
#         shutil.copy(input.runfile_nvt,output.runfile_nvt)
#         shutil.copy(input.runfile_production,output.runfile_production)
#         shutil.copy(input.index,output.index)



# rule copy_gromacs_monomer_input:
#     """
#     Copy the rest of necessary files to start a sqra run with a GROMACS calculation and potentially adapt default mdrun params.
#     """
#     input:
#         m1_topology = f"<inputs_gromacs>topol_m{{i}}.top",
#     output:
#         m1_topology= f"<outputs>molecule{{i}}/topol_m{{i}}.top",
#     run:
#         import shutil
#         shutil.copy(input.m1_topology,output.m1_topology)

# rule gromacs_monomer_energy:
#     wildcard_constraints:
#         ENERGY_PROGRAM  = "GROMACS"
#     input:
#         structure = f"<outputs>{NAME_PT_FOLDER}molecule{{i}}.<ext_str>",
#         runfile = f"<outputs>molecule{{i}}/mdrun.mdp",
#         topology = f"<outputs>molecule{{i}}/topol_m{{i}}.top",
#         select_energy = f"<outputs>molecule{{i}}/select_energy",
#         force_field_stuff = f"<outputs>molecule{{i}}/force_field_stuff/"
#     #shadow: "minimal"
#     log:
#         log = f"<outputs>molecule{{i}}/logging_gromacs.log"
#     benchmark:
#         repeat(f"<outputs>molecule{{i}}/gromacs_benchmark.txt", 1)
#     output:
#         energy = f"<outputs>molecule{{i}}/energy.xvg",
#     shell:
#         """
#         initial_dir=$(pwd)
#         cd $(dirname {input.runfile})
#         export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
#         gmx22 grompp -f $(basename {input.runfile}) -c ../pseudotrajectory/$(basename {input.structure}) -p $(basename {input.topology}) -o result.tpr  -maxwarn 1
#         gmx22 mdrun -s result.tpr -rerun ../pseudotrajectory/$(basename {input.structure}) -g $(basename {log.log})
#         gmx22 energy -f ener.edr -o $(basename {output.energy}) < $(basename {input.select_energy})
#         cd "$initial_dir" || exit
#         """





# rule gromacs_monomer_equilibration:
#     wildcard_constraints:
#         ENERGY_PROGRAM  = "GROMACS"
#     input:
#         structure = f"<outputs>molecule{{i}}/molecule{{i}}.<ext_str>",
#         runfile_minim=f"<outputs>molecule{{i}}/minim.mdp",
#         runfile_nvt=f"<outputs>molecule{{i}}/nvt.mdp",
#         topology = f"<outputs>molecule{{i}}/topol_m{{i}}.top",
#         force_field_stuff = f"<outputs>molecule{{i}}/force_field_stuff/"
#     output:
#         nvt = f"<outputs>molecule{{i}}/nvt.gro",
#     shell:
#         """
#         initial_dir=$(pwd)
#         cd $(dirname {input.runfile_minim})
#         export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
#         gmx22 grompp -f $(basename {input.runfile_minim}) -c $(basename {input.structure}) -p $(basename {input.topology}) -o em.tpr -r $(basename {input.structure}) -maxwarn 1
#         gmx22 mdrun -v -deffnm em
#
#         gmx22 grompp -f $(basename {input.runfile_nvt}) -c em.gro -r em.gro -p $(basename {input.topology}) -o nvt.tpr -maxwarn 1
#         gmx22 mdrun -v -deffnm nvt
#
#         cd "$initial_dir" || exit
#         """
#
# rule gromacs_monomer_production:
#     wildcard_constraints:
#         ENERGY_PROGRAM="GROMACS"
#     input:
#         structure=f"<outputs>molecule{{i}}/nvt.gro",
#         runfile=f"<outputs>molecule{{i}}/production.mdp",
#         topology=f"<outputs>molecule{{i}}/topol_m{{i}}.top",
#         select_energy=f"<outputs>molecule{{i}}/select_energy",
#         force_field_stuff=f"<outputs>molecule{{i}}/force_field_stuff/"
#     output:
#         energy=f"<outputs>molecule{{i}}/full_energy.xvg",
#         trajectory=f"<outputs>molecule{{i}}/trajectory.xtc",
#     shell:
#         """
#         initial_dir=$(pwd)
#         cd $(dirname {input.runfile})
#         export PATH="/home/janjoswig/local/gromacs-2022/bin:$PATH"
#         gmx22 grompp -f $(basename {input.runfile}) -c $(basename {input.structure}) -p $(basename {input.topology}) -o trajectory.tpr -maxwarn 1
#         gmx22 mdrun -v -deffnm trajectory
#         gmx22 energy -f trajectory.edr -o $(basename {output.energy}) < $(basename {input.select_energy})
#         cd "$initial_dir" || exit
#         """