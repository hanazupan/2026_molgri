from pathlib import Path
from workflow.helpers.orca_reader import QuantumSetup, OrcaReader, QuantumMolecule, OrcaWriter, read_important_stuff_into_csv, nice_str_of, split_xyz_trajectory

from pathlib import Path

def get_frames(wildcards):
    split_dir = checkpoints.split_trajectory.get().output[0]
    return [p.name for p in Path(split_dir).iterdir() if p.is_dir()]

rule read_orca:
    input:
        orca_out="/home/nadjar02/MA/2026_molgri/nobackup/benzene_benzene/test/sp.out"

    run:
        reader = OrcaReader(input.orca_out)

        energy = reader.extract_energies_orca_output()
        print("Energy:", energy)

        time = reader.extract_time_orca_output()
        print("Time:", time)

        print("Finished:", reader.assert_normal_finish(False))


BASE_DIR = Path("/home/nadjar02/MA/2026_molgri/nobackup/benzene_benzene/test")
FRAMES = glob_wildcards(
    "/home/nadjar02/MA/2026_molgri/nobackup/benzene_benzene/test/{frame}/sp.out"
).frame
OUT_PATH = BASE_DIR / "output.csv"

setup = QuantumSetup(
    functional="PBE0",
    basis_set="def2-TZVP",
    solvent=None,
    dispersion_correction="D3",
    num_scf=None,
    num_cores="4",
    ram_per_core="300"
)

rule write_csv:
    input:
        expand(
            "/home/nadjar02/MA/2026_molgri/nobackup/benzene_benzene/test/{frame}/sp.out",
            frame=FRAMES
        )
    output:
        csv="/home/nadjar02/MA/2026_molgri/nobackup/benzene_benzene/test/output.csv"
    params:
        setup=setup
    run:
        read_important_stuff_into_csv(
            out_files_to_read=input,
            csv_file_to_write=output.csv,
            setup=params.setup,
            num_points=len(input)
        )

rule write_orca_input:
    input:
        xyz=lambda wc: f"/home/nadjar02/MA/2026_molgri/nobackup/benzene_benzene/test/split/{wc.frame}/structure.xyz"
    output:
        inp="/home/nadjar02/MA/2026_molgri/nobackup/benzene_benzene/test/split/{frame}/structure.inp"
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

checkpoint split_trajectory:
    input:
        xyz="/home/nadjar02/MA/2026_molgri/nobackup/benzene_benzene/test/trajectory.xyz"
    output:
        directory("/home/nadjar02/MA/2026_molgri/nobackup/benzene_benzene/test/split")
    run:
        split_xyz_trajectory(
            xyz_file=input.xyz,
            output_base_dir=output[0],
            structures_per_chunk=5
        )
        # just create a flag so Snakemake knows we're done
        #Path(output[0]).mkdir(exist_ok=True)