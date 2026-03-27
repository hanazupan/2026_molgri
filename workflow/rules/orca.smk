from workflow.helpers.orca_reader import QuantumSetup, OrcaReader, QuantumMolecule, OrcaWriter, filter_frame_indices, write_energies_with_indices
from workflow.helpers.orca_reader import read_important_stuff_into_csv, read_energies_into_csv, nice_str_of, split_xyz_trajectory, xtc_to_xyz, find_invalid_frames_with_overlapping_atoms, extract_frame_indices_from_xyz
from workflow.helpers.remove_overlapping_cooridnates import remove_coordinates
import numpy as np
from pathlib import Path

# path on curta
REMOTE_BASE_DIR = "/home/nadjar02/MA/benzene"
REMOTE_TEST_DIR = f"{REMOTE_BASE_DIR}/cart_20_7_7_4"
#path on qcm
LOCAL_TEST_DIR = "/home/nadjar02/MA/2026_molgri/nobackup/benzene_benzene/cart_20_7_7_4"
GRID_DIR = "/home/nadjar02/MA/2026_molgri/nobackup/benzene_benzene/cart_20_7_7_4/pseudosimulation"

CHUNK_SIZE = 280

#FRAMES = glob_wildcards( "/home/nadjar02/MA/2026_molgri/nobackup/benzene_benzene/spherical_grid_20_42_4/{frame}/structure.out" ).frame

setup = QuantumSetup(
    functional="PBE0",
    basis_set="def2-TZVP",
    solvent=None,
    dispersion_correction="D3",
    num_scf=None,
    num_cores=4,        # ← IMPORTANT
    ram_per_core=400      # ← IMPORTANT
)

rule xtc_to_xyz:
    input:
        xtc = f"<pseudosimulation>trajectory.xtc",
        gro = f"<pseudosimulation>structure.gro"
    output:
        xyz = f"{GRID_DIR}/trajectory.xyz"
    run:
        xtc_to_xyz(
            input.xtc,
            input.gro,
            output.xyz
        )

rule find_invalid_frames:
    input:
        trajectory = "<pseudosimulation>trajectory.xyz"
    output:
        valid = "<outputs_indices>valid_indices.npy",
        invalid = "<outputs_indices>invalid_indices.npy"
    run:
        valid_indices, invalid_indices = find_invalid_frames_with_overlapping_atoms(
            input.trajectory
        )

        np.save(output.valid, valid_indices)
        np.save(output.invalid, invalid_indices)

rule copy_trajectory:
    input:
        f"{GRID_DIR}/trajectory.xyz"
    output:
        f"{LOCAL_TEST_DIR}/trajectory.xyz"
    shell:
        """
        echo "copy trajectory"
        cp {input} {output}
        """

rule clean_trajectory:
    input:
        xyz=f"{LOCAL_TEST_DIR}/trajectory.xyz"
    output:
        xyz=f"{LOCAL_TEST_DIR}/cleaned_trajectory.xyz"
    params:
        tol=1e-6
    run:
        remove_coordinates(
            input_file=input.xyz,
            output_file=output.xyz,
            tol=params.tol
        )

checkpoint split_trajectory:
    input:
        xyz=f"{LOCAL_TEST_DIR}/cleaned_trajectory.xyz"
    output:
        flag=f"{LOCAL_TEST_DIR}/split_done.txt"
    run:
        split_xyz_trajectory(
            xyz_file=input.xyz,
            output_base_dir=f"{LOCAL_TEST_DIR}",
            structures_per_chunk= CHUNK_SIZE
        )

        with open(output.flag, "w") as f:
            f.write("done")
        # just create a flag so Snakemake knows we're done
        #Path(output[0]).mkdir(exist_ok=True)

def get_frames(wildcards):
    checkpoints.split_trajectory.get()

    base_dir = Path(LOCAL_TEST_DIR)

    return sorted([
        p.name for p in base_dir.iterdir()
        if p.is_dir() and p.name.isdigit()
    ])

# import os
# def get_frames():
#     return sorted([
#         d for d in os.listdir(LOCAL_TEST_DIR)
#         if d.isdigit()
#     ])

rule write_orca_input:
    input:
        xyz=lambda wc: f"{LOCAL_TEST_DIR}/{wc.frame}/structure.xyz"
    output:
        inp=f"{LOCAL_TEST_DIR}/{{frame}}/structure.inp"
    params:
        setup=setup
    run:
        molecule = QuantumMolecule(
            xyz_file=input.xyz,
            charge=0,
            multiplicity=1
        )

        writer = OrcaWriter(molecule, params.setup)
        writer.make_input(geo_optimization=False)
        writer.write_to_file(output.inp)



rule copy_to_curta:
    input:
        lambda wc: expand(
            f"{LOCAL_TEST_DIR}/{{frame}}/structure.inp",
            frame=get_frames(wc)
        )
    output:
        touch(f"{LOCAL_TEST_DIR}/copied_to_curta.txt")
    shell:
        f"""
        echo "=== Copying to Curta ==="
        ssh curta "mkdir -p {REMOTE_BASE_DIR}"
        rsync -av {LOCAL_TEST_DIR}/ curta:{REMOTE_TEST_DIR}/
        touch {output}
        """

rule run_orca_curta:
    input:
        f"{LOCAL_TEST_DIR}/copied_to_curta.txt"
    output:
        touch(f"{LOCAL_TEST_DIR}/curta_started.txt")
    shell:
        f"""
        echo "=== Running ORCA on Curta ==="
        ssh curta "cd {REMOTE_TEST_DIR} && bash ~/run/submit_on_curta.sh"
        touch {output}
        """

rule copy_back_from_curta:
    input:
        f"{LOCAL_TEST_DIR}/curta_started.txt"
    output:
        touch(f"{LOCAL_TEST_DIR}/copied_back.txt")
    shell:
        f"""
        echo "=== Copying results back ==="
        rsync -av curta:{REMOTE_TEST_DIR}/ {LOCAL_TEST_DIR}/
        touch {output}
        """

# rule run_orca_locally:
#     input:
#         inp=f"{LOCAL_TEST_DIR}/{{frame}}/structure.inp"
#     output:
#         out=f"{LOCAL_TEST_DIR}/{{frame}}/structure.out"
#     log:
#         f"{LOCAL_TEST_DIR}/{{frame}}/orca.log"
#     resources:
#         orca=1
#     shell:
#         """
#         cd $(dirname {input.inp})
#         orca structure.inp > {log} 2>&1
#         """

rule read_orca:
    input:
        orca_out= lambda wildcards: expand(
            f"{LOCAL_TEST_DIR}/{{frame}}/structure.out",
            frame=get_frames()
        )

    run:
        reader = OrcaReader(input.orca_out)

        energy = reader.extract_energies_orca_output()
        print("Energy:", energy)

        time = reader.extract_time_orca_output()
        print("Time:", time)

        print("Finished:", reader.assert_normal_finish(False))

# rule write_csv:
#     input:
#         expand(
#             f"{LOCAL_TEST_DIR}/{frame}/structure.out",
#             frame=FRAMES
#         )
#     output:
#         csv=f"{LOCAL_TEST_DIR}/output.csv"
#     params:
#         setup=setup
#     run:
#         read_important_stuff_into_csv(
#             out_files_to_read=input,
#             csv_file_to_write=output.csv,
#             setup=params.setup,
#             num_points=len(input)
#         )

rule write_large_csv:
    input:
        lambda wc: expand(
            f"{LOCAL_TEST_DIR}/{{frame}}/structure.out",
            frame=get_frames(wc)
        )
    output:
        csv=f"{LOCAL_TEST_DIR}/output.csv"
    run:
        read_important_stuff_into_csv(
            out_files_to_read=input,
            csv_file_to_write=output.csv,
            setup=setup,
            num_points=len(input),
            chunksize=CHUNK_SIZE
        )


rule write_small_csv:
    input:
        traj=f"{LOCAL_TEST_DIR}/trajectory.xyz",
        outs=lambda wc: expand(
            f"{LOCAL_TEST_DIR}/{{frame}}/structure.out",
            frame=get_frames(wc)
        )
    output:
        csv=f"{LOCAL_TEST_DIR}/energy.csv"
    run:
        write_energies_with_indices(
            out_files_to_read=input.outs,
            trajectory_path=input.traj,
            csv_file_to_write=output.csv
        )